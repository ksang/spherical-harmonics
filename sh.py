
import math
import argparse
parser = argparse.ArgumentParser(description='Spherical Harmonics')
parser.add_argument('-l', '--degree', type=int, default=3, help='Degree')
parser.add_argument('--theta', type=float, default=0.0, help='Theta in radians')
parser.add_argument('--phi', type=float, default=0.0, help='Phi in radians')

# Renormalisation constant for SH function
def K(l: int, m: int) -> float:
    temp = ((2.0*l + 1.0)*math.factorial(l-m)) / (4.0*math.pi*math.factorial(l+m))
    return math.sqrt(temp)

def P(l: int, m: int, x: float) -> float:
    pmm = 1.0
    if m > 0:
        somx2 = math.sqrt((1.0-x))
        fact = 1.0;
        for _ in range(m):
            pmm *= (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    
    pmmp1 = x * (2.0*m+1.0) * pmm
    if l == m+1:
        return pmmp1;
    
    pll = 0.0;
    for ll in range(m+2, l+1):
        pll =  ((2.0*ll-1.0)*x*pmmp1-(ll+m-1.0)*pmm ) / (ll-m)
        pmm = pmmp1
        pmmp1 = pll

    return pll;

def SphericalHarmonics(l: int, m: int, theta: float, phi: float):
    sqrt2 = math.sqrt(2.0);
    if m == 0:
        return K(l,0)*P(l,m,math.cos(theta));
    elif m > 0:
        return sqrt2*K(l,m)*math.cos(m*phi)*P(l,m,math.cos(theta));
    else:
        return sqrt2*K(l,-m)*math.sin(-m*phi)*P(l,-m,math.cos(theta));


if __name__ == '__main__':
    args = parser.parse_args()
    l = args.order
    theta = args.theta
    phi = args.phi
    print("theta: {}, phi: {}".format(theta, phi))
    for l in range(l+1):
        for m in range(-l, l+1):
            v = SphericalHarmonics(l, m, theta, phi)
            print("l: {}, \tm: {}, \tSH: {}".format(l, m, v))