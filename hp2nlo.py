import sympy as sp
from sympy import cos, sin, symbols, Matrix, simplify, tensorcontraction, tensorproduct, integrate, pi, legendre
import numpy as np
import re
from typing import Dict, Optional, List

def b_lab_avg(T_mol, i, j, k, delta, phi, RR, symmetry_dict):
    """
    rotate mol frame to lab
    average over delta and phi
    do symmetry substitutions and simplify
    """
    element = 0
    for l in range(3):
        for m in range(3):
            for n in range(3):
                element += RR[i,l] * RR[j,m] * RR[k,n] * T_mol[l,m,n]

    result = integrate(element, (delta, 0, 2*pi)) / (2 * pi)
    result2 = integrate(result, (phi, 0, 2*pi)) / (2 * pi)

    b_lab_avg_simplified = simplify(result2.subs(symmetry_dict))
    return b_lab_avg_simplified

def ODF(x, P1, P2, P3, P4):
    "orientational distribution funciton"
    return (1/2 +
            (3/2) * P1 * legendre(1, x) + 
            (5/2) * P2 * legendre(2, x) +
            (7/2) * P3 * legendre(3, x) +
            (9/2) * P4 * legendre(4, x))

def integrate_with_odf(b_lab_avg, theta, P1, P2, P3, P4):
    " integrate to get average"
    result = b_lab_avg * ODF(cos(theta), P1, P2, P3, P4) * sin(theta)
    return simplify(integrate(result, (theta, 0, pi)))

def extract_hyperpolarizability(
    filename: str,
    wavelength: float,
    beta_type: str = "Beta(-2w;w,w)",
    orientation: str = "input",
    debug: bool = False
) -> Dict[str, float]:
    """
    Extract hyperpolarizability tensor values from Gaussian output file.
    
    Parameters:
    -----------
    filename : str
        Path to the Gaussian output file
    wavelength : float
        Wavelength in nm to extract (e.g., 800.0)
    beta_type : str
        Type of Beta calculation (e.g., "Beta(0;0,0)", "Beta(-w;w,0)", "Beta(-2w;w,w)")
        Default: "Beta(-2w;w,w)"
    orientation : str
        Either "input" or "dipole" orientation (default: "input")
    debug : bool
        Print debug information (default: False)
    
    Returns:
    --------
    Dict[str, float]
        Dictionary with tensor components as keys and SI values as floats
    """
    
    with open(filename, 'r') as f:
        content = f.read()
    
    if debug:
        print(f"File length: {len(content)} characters")
    
    # Find the start of the target orientation section
    # Pattern matches: "First dipole hyperpolarizability, Beta (input orientation)."
    orientation_header = f"First dipole hyperpolarizability, Beta ({orientation} orientation)."
    
    # Find all occurrences of this header
    header_positions = []
    start = 0
    while True:
        pos = content.find(orientation_header, start)
        if pos == -1:
            break
        header_positions.append(pos)
        start = pos + 1
    
    if debug:
        print(f"Found {len(header_positions)} '{orientation} orientation' headers at positions: {header_positions}")
    
    if not header_positions:
        raise ValueError(f"No {orientation} orientation section found")
    
    
    section_start = header_positions[-1]
    
    if debug:
        print(f"Using LAST header at position: {section_start}")
    
    # Find the end of this section
    # End at "Second dipole hyperpolarizability" for this orientation
    # or at "First dipole hyperpolarizability" for the OTHER orientation
    other_orientation = "dipole" if orientation == "input" else "input"
    
    end_markers = [
        f'Second dipole hyperpolarizability, Gamma ({orientation} orientation).',
        f'First dipole hyperpolarizability, Beta ({other_orientation} orientation).',
        'Dipole orientation:',
    ]
    
    section_end = len(content)
    for marker in end_markers:
        pos = content.find(marker, section_start + len(orientation_header))
        if pos != -1 and pos < section_end:
            section_end = pos
            if debug:
                print(f"Section ends at '{marker[:50]}...' position {pos}")
    
    target_section = content[section_start:section_end]
    
    if debug:
        print(f"Section from {section_start} to {section_end} ({len(target_section)} chars)")
        # Show what beta blocks exist in the section
        beta_matches = re.findall(r'Beta\([^)]+\)[^:]*:', target_section)
        print(f"Beta blocks found in section: {beta_matches}")
    
    # Now find the specific beta type and wavelength within this section
    beta_type_escaped = re.escape(beta_type)
    
    # For static case (Beta(0;0,0)), no wavelength specification
    if "0;0,0" in beta_type:
        # Match Beta(0;0,0): followed by data until next Beta( or end
        pattern = rf"{beta_type_escaped}:\s*\n(.*?)(?=\n\s*Beta\(|\Z)"
    else:
        # For frequency-dependent cases, match beta type AND wavelength
        # Handle variable spacing: "w= 800.0nm" or "w=  800.0nm" or "w=800.0nm"
        pattern = rf"{beta_type_escaped}\s+w=\s*{wavelength}nm:\s*\n(.*?)(?=\n\s*Beta\(|\n\s*Second|\Z)"
    
    if debug:
        print(f"\nSearching with pattern for: {beta_type} w={wavelength}nm")
    
    match = re.search(pattern, target_section, re.DOTALL)
    
    if not match:
        if debug:
            # Try to find any occurrence of the beta type to diagnose
            simple_pattern = rf"{beta_type_escaped}"
            simple_matches = list(re.finditer(simple_pattern, target_section))
            print(f"\nSimple search for '{beta_type}' found {len(simple_matches)} matches")
            for m in simple_matches[:3]:
                context = target_section[m.start():m.start()+100]
                print(f"  Context: {repr(context)}")
        raise ValueError(f"No {beta_type} data found for wavelength {wavelength} nm in {orientation} orientation")
    
    target_block = match.group(1)
    
    if debug:
        print(f"\nExtracted block ({len(target_block)} chars):")
        print(target_block[:500] if len(target_block) > 500 else target_block)
    
    # Extract tensor values
    tensor_dict = {}
    
    # Pattern to match lines with tensor components
    # Component names: xxx, xxy, yxy, yyy, etc. or || (z), _|_(z), ||, x, y, z
    # The line format is:
    #    component     au_value           esu_value          si_value
    line_pattern = r'^\s+([xyz]{3}|[xyz]{2}|[xyz]|\|\|\s*\(z\)|\|\||\s*_\|_\(z\))\s+([+-]?\d+\.\d+D[+-]\d+)\s+([+-]?\d+\.\d+D[+-]\d+)\s+([+-]?\d+\.\d+D[+-]\d+)'
    
    lines = target_block.split('\n')
    for line in lines:
        match = re.match(line_pattern, line)
        if match:
            component = match.group(1).strip()
            si_value_str = match.group(4)
            
            # Convert Fortran notation (D) to Python notation (E)
            si_value_str = si_value_str.replace('D', 'E')
            si_value_float = float(si_value_str)
            
            tensor_dict[component] = si_value_float
    
    if debug:
        print(f"\nExtracted {len(tensor_dict)} components")
    
    return tensor_dict


def print_tensor_dict(tensor_dict: Dict[str, float]) -> None:
    """Pretty print the tensor dictionary."""
    print("Hyperpolarizability Tensor (SI units):")
    print("-" * 50)
    for key, value in tensor_dict.items():
        print(f'"{key}": {value:.6e}')


def convert_hp_2_nlo(tensor, density=1280, mol_mass = 424, p1_val = 0.723, p3_val= -0.14 ):
    # sympy define symbols and instantiate euler matrix
    # sympy define symbols and instantiate euler matrix
    phi, theta, delta = symbols('phi theta delta', real=True)
    # legendre
    P1, P2, P3, P4 = symbols('P1 P2 P3 P4', real = True)

    Rz1 = Matrix([
        [cos(phi), -sin(phi), 0],
        [sin(phi), cos(phi), 0],
        [0, 0, 1]
    ])


    # Rotation around Y by theta
    Ry = Matrix([
        [cos(theta), 0, sin(theta)],
        [0, 1, 0],
        [-sin(theta), 0, cos(theta)]
    ])

    # Third rotation around Z by delta
    Rz2 = Matrix([
        [cos(delta), -sin(delta), 0],
        [sin(delta), cos(delta), 0],
        [0, 0, 1]
    ])

    RR = Rz1 * Ry * Rz2

    # instantiate T_mol
    T_mol = sp.MutableDenseNDimArray.zeros(3,3,3)

    for l in range(3):
        for m in range(3):
            for n in range(3):
                T_mol[l, m, n] = symbols(f'T_{l}{m}{n}')

    symmetry_dict = {
        symbols('T_020'): symbols('T_002'),
        symbols('T_121'): symbols('T_112'),
        symbols('T_021'): symbols('T_012'),
        symbols('T_120'): symbols('T_102'),
        symbols('T_201'): symbols('T_210')
    }

    b_lab_avg_222 = b_lab_avg(T_mol, 2,2,2,delta, phi, RR, symmetry_dict)
    b_lab_avg_200 = b_lab_avg(T_mol, 2, 0, 0, delta, phi, RR, symmetry_dict)
    b_lab_avg_020 = b_lab_avg(T_mol, 0, 2, 0,delta, phi, RR,  symmetry_dict)
    b222th = integrate_with_odf(b_lab_avg_222, theta, P1, P2, P3, P4)
    b200th = integrate_with_odf(b_lab_avg_200, theta,  P1, P2, P3, P4)
    b020th = integrate_with_odf(b_lab_avg_020, theta,  P1, P2, P3, P4)

    # i think these substitutions come from molecular long axis arguments?
    hp_substitution_list = {
        symbols('T_002') : tensor["yyx"],
        symbols('T_222') : tensor["xxx"],
        symbols('T_112') : tensor["zzx"],
        symbols('T_200') : tensor["xyy"],
        symbols('T_211') : tensor["xzz"]
    }

    b222th = b222th.subs(hp_substitution_list)
    b200th = b200th.subs(hp_substitution_list)  
    b020th = b020th.subs(hp_substitution_list)    


    #  experimental components 
    Na = 6.02e26 #avo

    NN = Na * density / mol_mass # number density

    nn1 = 1.72
    nn2 = 1.52

    ff1 = (nn1**2+2) / 3
    ff2 = (nn2**2+2) / 3

    fact1 = (ff1**3 * NN *10**-50) / (2 * 8.85e-12)
    fact2 = (ff2**3 * NN *10**-50) / (2 * 8.85e-12)

    order_param_subs = {P1: p1_val, P3: p3_val}
    b222th_numeric = b222th.subs(order_param_subs)
    b200th_numeric = b200th.subs(order_param_subs)
    b020th_numeric = b020th.subs(order_param_subs)

    d_222 = fact1 * b222th_numeric
    d_200 = fact2 * b200th_numeric
    d_020 = fact1 * b020th_numeric

    print(f"d33 (pm/V) = {float(d_222):.5e}")
    print(f"d31 (pm/V) = {float(d_200):.5e}")
    print(f"d15 (pm/V) = {float(d_020):.5e}")

    return d_222, d_200, d_020


