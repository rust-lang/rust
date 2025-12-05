//! 4x4 matrix inverse
// Code ported from the `packed_simd` crate
// Run this code with `cargo test --example matrix_inversion`
#![feature(array_chunks, portable_simd)]
use core_simd::simd::prelude::*;

// Gotta define our own 4x4 matrix since Rust doesn't ship multidim arrays yet :^)
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Matrix4x4([[f32; 4]; 4]);

#[allow(clippy::too_many_lines)]
pub fn scalar_inv4x4(m: Matrix4x4) -> Option<Matrix4x4> {
    let m = m.0;

    #[rustfmt::skip]
    let mut inv = [
        // row 0:
        [
            // 0,0:
            m[1][1] * m[2][2] * m[3][3] -
            m[1][1] * m[2][3] * m[3][2] -
            m[2][1] * m[1][2] * m[3][3] +
            m[2][1] * m[1][3] * m[3][2] +
            m[3][1] * m[1][2] * m[2][3] -
            m[3][1] * m[1][3] * m[2][2],
            // 0,1:
           -m[0][1] * m[2][2] * m[3][3] +
            m[0][1] * m[2][3] * m[3][2] +
            m[2][1] * m[0][2] * m[3][3] -
            m[2][1] * m[0][3] * m[3][2] -
            m[3][1] * m[0][2] * m[2][3] +
            m[3][1] * m[0][3] * m[2][2],
            // 0,2:
            m[0][1] * m[1][2] * m[3][3] -
            m[0][1] * m[1][3] * m[3][2] -
            m[1][1] * m[0][2] * m[3][3] +
            m[1][1] * m[0][3] * m[3][2] +
            m[3][1] * m[0][2] * m[1][3] -
            m[3][1] * m[0][3] * m[1][2],
            // 0,3:
           -m[0][1] * m[1][2] * m[2][3] +
            m[0][1] * m[1][3] * m[2][2] +
            m[1][1] * m[0][2] * m[2][3] -
            m[1][1] * m[0][3] * m[2][2] -
            m[2][1] * m[0][2] * m[1][3] +
            m[2][1] * m[0][3] * m[1][2],
        ],
        // row 1
        [
            // 1,0:
           -m[1][0] * m[2][2] * m[3][3] +
            m[1][0] * m[2][3] * m[3][2] +
            m[2][0] * m[1][2] * m[3][3] -
            m[2][0] * m[1][3] * m[3][2] -
            m[3][0] * m[1][2] * m[2][3] +
            m[3][0] * m[1][3] * m[2][2],
            // 1,1:
            m[0][0] * m[2][2] * m[3][3] -
            m[0][0] * m[2][3] * m[3][2] -
            m[2][0] * m[0][2] * m[3][3] +
            m[2][0] * m[0][3] * m[3][2] +
            m[3][0] * m[0][2] * m[2][3] -
            m[3][0] * m[0][3] * m[2][2],
            // 1,2:
           -m[0][0] * m[1][2] * m[3][3] +
            m[0][0] * m[1][3] * m[3][2] +
            m[1][0] * m[0][2] * m[3][3] -
            m[1][0] * m[0][3] * m[3][2] -
            m[3][0] * m[0][2] * m[1][3] +
            m[3][0] * m[0][3] * m[1][2],
            // 1,3:
            m[0][0] * m[1][2] * m[2][3] -
            m[0][0] * m[1][3] * m[2][2] -
            m[1][0] * m[0][2] * m[2][3] +
            m[1][0] * m[0][3] * m[2][2] +
            m[2][0] * m[0][2] * m[1][3] -
            m[2][0] * m[0][3] * m[1][2],
        ],
        // row 2
        [
            // 2,0:
            m[1][0] * m[2][1] * m[3][3] -
            m[1][0] * m[2][3] * m[3][1] -
            m[2][0] * m[1][1] * m[3][3] +
            m[2][0] * m[1][3] * m[3][1] +
            m[3][0] * m[1][1] * m[2][3] -
            m[3][0] * m[1][3] * m[2][1],
            // 2,1:
           -m[0][0] * m[2][1] * m[3][3] +
            m[0][0] * m[2][3] * m[3][1] +
            m[2][0] * m[0][1] * m[3][3] -
            m[2][0] * m[0][3] * m[3][1] -
            m[3][0] * m[0][1] * m[2][3] +
            m[3][0] * m[0][3] * m[2][1],
            // 2,2:
            m[0][0] * m[1][1] * m[3][3] -
            m[0][0] * m[1][3] * m[3][1] -
            m[1][0] * m[0][1] * m[3][3] +
            m[1][0] * m[0][3] * m[3][1] +
            m[3][0] * m[0][1] * m[1][3] -
            m[3][0] * m[0][3] * m[1][1],
            // 2,3:
           -m[0][0] * m[1][1] * m[2][3] +
            m[0][0] * m[1][3] * m[2][1] +
            m[1][0] * m[0][1] * m[2][3] -
            m[1][0] * m[0][3] * m[2][1] -
            m[2][0] * m[0][1] * m[1][3] +
            m[2][0] * m[0][3] * m[1][1],
        ],
        // row 3
        [
            // 3,0:
           -m[1][0] * m[2][1] * m[3][2] +
            m[1][0] * m[2][2] * m[3][1] +
            m[2][0] * m[1][1] * m[3][2] -
            m[2][0] * m[1][2] * m[3][1] -
            m[3][0] * m[1][1] * m[2][2] +
            m[3][0] * m[1][2] * m[2][1],
            // 3,1:
            m[0][0] * m[2][1] * m[3][2] -
            m[0][0] * m[2][2] * m[3][1] -
            m[2][0] * m[0][1] * m[3][2] +
            m[2][0] * m[0][2] * m[3][1] +
            m[3][0] * m[0][1] * m[2][2] -
            m[3][0] * m[0][2] * m[2][1],
            // 3,2:
           -m[0][0] * m[1][1] * m[3][2] +
            m[0][0] * m[1][2] * m[3][1] +
            m[1][0] * m[0][1] * m[3][2] -
            m[1][0] * m[0][2] * m[3][1] -
            m[3][0] * m[0][1] * m[1][2] +
            m[3][0] * m[0][2] * m[1][1],
            // 3,3:
            m[0][0] * m[1][1] * m[2][2] -
            m[0][0] * m[1][2] * m[2][1] -
            m[1][0] * m[0][1] * m[2][2] +
            m[1][0] * m[0][2] * m[2][1] +
            m[2][0] * m[0][1] * m[1][2] -
            m[2][0] * m[0][2] * m[1][1],
        ],
    ];

    let det = m[0][0] * inv[0][0] + m[0][1] * inv[1][0] + m[0][2] * inv[2][0] + m[0][3] * inv[3][0];
    if det == 0. {
        return None;
    }

    let det_inv = 1. / det;

    for row in &mut inv {
        for elem in row.iter_mut() {
            *elem *= det_inv;
        }
    }

    Some(Matrix4x4(inv))
}

pub fn simd_inv4x4(m: Matrix4x4) -> Option<Matrix4x4> {
    let m = m.0;
    let m_0 = f32x4::from_array(m[0]);
    let m_1 = f32x4::from_array(m[1]);
    let m_2 = f32x4::from_array(m[2]);
    let m_3 = f32x4::from_array(m[3]);

    const SHUFFLE01: [usize; 4] = [0, 1, 4, 5];
    const SHUFFLE02: [usize; 4] = [0, 2, 4, 6];
    const SHUFFLE13: [usize; 4] = [1, 3, 5, 7];
    const SHUFFLE23: [usize; 4] = [2, 3, 6, 7];

    let tmp = simd_swizzle!(m_0, m_1, SHUFFLE01);
    let row1 = simd_swizzle!(m_2, m_3, SHUFFLE01);

    let row0 = simd_swizzle!(tmp, row1, SHUFFLE02);
    let row1 = simd_swizzle!(row1, tmp, SHUFFLE13);

    let tmp = simd_swizzle!(m_0, m_1, SHUFFLE23);
    let row3 = simd_swizzle!(m_2, m_3, SHUFFLE23);
    let row2 = simd_swizzle!(tmp, row3, SHUFFLE02);
    let row3 = simd_swizzle!(row3, tmp, SHUFFLE13);

    let tmp = (row2 * row3).reverse().rotate_elements_right::<2>();
    let minor0 = row1 * tmp;
    let minor1 = row0 * tmp;
    let tmp = tmp.rotate_elements_right::<2>();
    let minor0 = (row1 * tmp) - minor0;
    let minor1 = (row0 * tmp) - minor1;
    let minor1 = minor1.rotate_elements_right::<2>();

    let tmp = (row1 * row2).reverse().rotate_elements_right::<2>();
    let minor0 = (row3 * tmp) + minor0;
    let minor3 = row0 * tmp;
    let tmp = tmp.rotate_elements_right::<2>();

    let minor0 = minor0 - row3 * tmp;
    let minor3 = row0 * tmp - minor3;
    let minor3 = minor3.rotate_elements_right::<2>();

    let tmp = (row3 * row1.rotate_elements_right::<2>())
        .reverse()
        .rotate_elements_right::<2>();
    let row2 = row2.rotate_elements_right::<2>();
    let minor0 = row2 * tmp + minor0;
    let minor2 = row0 * tmp;
    let tmp = tmp.rotate_elements_right::<2>();
    let minor0 = minor0 - row2 * tmp;
    let minor2 = row0 * tmp - minor2;
    let minor2 = minor2.rotate_elements_right::<2>();

    let tmp = (row0 * row1).reverse().rotate_elements_right::<2>();
    let minor2 = minor2 + row3 * tmp;
    let minor3 = row2 * tmp - minor3;
    let tmp = tmp.rotate_elements_right::<2>();
    let minor2 = row3 * tmp - minor2;
    let minor3 = minor3 - row2 * tmp;

    let tmp = (row0 * row3).reverse().rotate_elements_right::<2>();
    let minor1 = minor1 - row2 * tmp;
    let minor2 = row1 * tmp + minor2;
    let tmp = tmp.rotate_elements_right::<2>();
    let minor1 = row2 * tmp + minor1;
    let minor2 = minor2 - row1 * tmp;

    let tmp = (row0 * row2).reverse().rotate_elements_right::<2>();
    let minor1 = row3 * tmp + minor1;
    let minor3 = minor3 - row1 * tmp;
    let tmp = tmp.rotate_elements_right::<2>();
    let minor1 = minor1 - row3 * tmp;
    let minor3 = row1 * tmp + minor3;

    let det = row0 * minor0;
    let det = det.rotate_elements_right::<2>() + det;
    let det = det.reverse().rotate_elements_right::<2>() + det;

    if det.reduce_sum() == 0. {
        return None;
    }
    // calculate the reciprocal
    let tmp = f32x4::splat(1.0) / det;
    let det = tmp + tmp - det * tmp * tmp;

    let res0 = minor0 * det;
    let res1 = minor1 * det;
    let res2 = minor2 * det;
    let res3 = minor3 * det;

    let mut m = m;

    m[0] = res0.to_array();
    m[1] = res1.to_array();
    m[2] = res2.to_array();
    m[3] = res3.to_array();

    Some(Matrix4x4(m))
}

#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use super::*;

    #[test]
    fn test() {
    let tests: &[(Matrix4x4, Option<Matrix4x4>)] = &[
        // Identity:
        (Matrix4x4([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
         ]),
         Some(Matrix4x4([
             [1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.],
         ]))
        ),
        // None:
        (Matrix4x4([
            [1., 2., 3., 4.],
            [12., 11., 10., 9.],
            [5., 6., 7., 8.],
            [16., 15., 14., 13.],
        ]),
         None
        ),
        // Other:
        (Matrix4x4([
            [1., 1., 1., 0.],
            [0., 3., 1., 2.],
            [2., 3., 1., 0.],
            [1., 0., 2., 1.],
        ]),
         Some(Matrix4x4([
             [-3., -0.5,   1.5,  1.0],
             [ 1., 0.25, -0.25, -0.5],
             [ 3., 0.25, -1.25, -0.5],
             [-3., 0.0,    1.0,  1.0],
         ]))
        ),


    ];

        for &(input, output) in tests {
            assert_eq!(scalar_inv4x4(input), output);
            assert_eq!(simd_inv4x4(input), output);
        }
    }
}

fn main() {
    // Empty main to make cargo happy
}
