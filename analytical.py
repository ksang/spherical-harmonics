import torch
import math
c0 = 0.28209479177387814  # Y[0,0] = 1/2 * math.sqrt(1/math.pi)
c1 = [
    0.4886025119029199,   # Y[1, -1] = 1/2 * math.sqrt(3/math.pi) * y
    0.4886025119029199,   # Y[1, 0] = 1/2 * math.sqrt(3/math.pi) * z
    0.4886025119029199,   # Y[1, 1] = 1/2 * math.sqrt(3/math.pi) * x
]
c2 = [
    1.0925484305920792,   # Y[2, -2] = 1/2 * math.sqrt(15/math.pi)* xy
    1.0925484305920792,   # Y[2, -1] = 1/2 * math.sqrt(15/math.pi)* yz
    0.31539156525252005,  # Y[1, 1] = 1/2 * math.sqrt(3/math.pi) * (3z^2-1)
    1.0925484305920792,   # Y[2, 1] = 1/2 * math.sqrt(15/math.pi) * xz
    0.5462742152960396,   # Y[2, 2] = 1/4 * math.sqrt(15/math.pi) * (x^2-y^2)
]
c3 = [
    0.5900435899266435,   # Y[3, -3] = 1/4 * math.sqrt(35/(2*math.pi)) * y(3x^2-y^2)
    2.890611442640554,    # Y[3, -2] = 1/2 * math.sqrt(105/(math.pi)) * xyz
    0.4570457994644658,   # Y[3, -1] = 1/4 * math.sqrt(21/(2*math.pi)) * y(5z^2-1)
    0.3731763325901154,   # Y[3, -1] = 1/4 * math.sqrt(7/(math.pi)) * (5z^3 - 3z)
    0.4570457994644658,   # Y[3, -1] = 1/4 * math.sqrt(21/(2*math.pi)) * x(5z^2-1)
    1.445305721320277,    # Y[3, 2] = 1/4 * math.sqrt(105/(math.pi)) * (x^2-y^2)z
    0.5900435899266435,   # Y[3, 3] = 1/4 * math.sqrt(35/(2*math.pi)) * x(x^2-3y^2)
]
c4 = [
    2.5033429417967046,   # Y[4, -4] = 3/4 * math.sqrt(35/(math.pi)) * xy(x^2-y^2)
    1.7701307697799304,   # Y[4, -3] = 3/4 * math.sqrt(35/(2*math.pi)) * y(3x^2-y^2)z
    0.9461746957575601,   # Y[4, -2] = 3/4 * math.sqrt(5/(math.pi)) * xy(7z^2-1)
    0.6690465435572892,   # Y[4, -1] = 3/4 * math.sqrt(5/(2*math.pi)) * y(7z^3-3z)
    0.10578554691520431,  # Y[4, 0] = 3/16 * math.sqrt(1/(math.pi)) * (35z^4-30z^2+3)
    0.6690465435572892,   # Y[4, 1] = 3/4 * math.sqrt(5/(2*math.pi)) * x(7z^3-3z)
    0.47308734787878004,  # Y[4, 2] = 3/8 * math.sqrt(5/(math.pi)) * (x^2-y^2)(7z^2-1)
    1.7701307697799304,   # Y[4, 3] = 3/4 * math.sqrt(35/(2*math.pi)) * x(x^2-3y^2)z
    0.6258357354491761,   # Y[4, 4] = 3/16 * math.sqrt(35/(math.pi)) * (x^2(x^2-3y^2)-y^2(3x^2-y^2))
]

def sh_encoding(xyz: torch.Tensor, l: int) -> torch.Tensor:
    """
    analytical implementation of spherial harmonics encoding for directional information
    expressions can be found from: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics

    Args:
        xyz (Tensor): input directional tensor in shape Bx3
        l (int): the order of shpherical harmonics, this analytical implementaion support up to 4

    returns:
        encoding (Tensor): spherical harmonics in shaoe Bx(l+1)^2
    """
    assert l >= 0 and l <=4, "order not supported"
    code = torch.empty((xyz.size(0), (l+1)**2), dtype=xyz.dtype, device=xyz.device)
    x, y, z = xyz.unbind(-1)
    if l >= 2:
        x2, y2, z2 = x*x, y*y, z*z
        xy, yz, xz = x*y, y*z, x*z
    for l in range(l+1):
        if l == 0:
            code[:,0] = c0
        elif l == 1:
            code[:,1] = c1[0]*y
            code[:,2] = c1[1]*z
            code[:,3] = c1[2]*x
        elif l == 2:
            code[:,4] = c2[0]*xy
            code[:,5] = c2[1]*yz
            code[:,6] = c2[2]*(3.0*z2-1.0)
            code[:,7] = c2[3]*xz
            code[:,8] = c2[4]*(x2-y2)
        elif l == 3:
            code[:,9] = c3[0]*y*(3.0*x2-y2)
            code[:,10] = c3[1]*xy*z
            code[:,11] = c3[2]*y*(5.0*z2-1.0)
            code[:,12] = c3[3]*(5.0*z2*z-3.0*z)
            code[:,13] = c3[4]*x*(5.0*z2-1.0)
            code[:,14] = c3[5]*(x2-y2)*z
            code[:,15] = c3[6]*x*(x2-3.0*y2)
        elif l == 4:
            code[:,16] = c4[0]*xy*(x2-y2)
            code[:,17] = c4[1]*yz*(3.0*x2-y2)
            code[:,18] = c4[2]*xy*(7.0*z2-1.0)
            code[:,19] = c4[3]*yz*(7.0*z2-3.0)
            code[:,20] = c4[4]*(z2*(35.0*z2-30.0)+3.0)
            code[:,21] = c4[5]*xz*(7.0*z2-3.0)
            code[:,22] = c4[6]*(x2-y2)*(7.0*z2-1.0)
            code[:,23] = c4[7]*xz*(x2-3.0*y2)
            code[:,24] = c4[8]*(x2*(x2-3.0*y2)-y2*(3.0*x2-y2))

    return code

if __name__ == '__main__':
    xyz = torch.rand((8, 3))
    code = sh_encoding(xyz, 4)
    print(xyz)
    print(code)