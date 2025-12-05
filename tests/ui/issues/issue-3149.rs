//@ check-pass
#![allow(dead_code)]
#![allow(non_snake_case)]

fn Matrix4<T>(m11: T, m12: T, m13: T, m14: T,
              m21: T, m22: T, m23: T, m24: T,
              m31: T, m32: T, m33: T, m34: T,
              m41: T, m42: T, m43: T, m44: T)
              -> Matrix4<T> {
    Matrix4 {
        m11: m11, m12: m12, m13: m13, m14: m14,
        m21: m21, m22: m22, m23: m23, m24: m24,
        m31: m31, m32: m32, m33: m33, m34: m34,
        m41: m41, m42: m42, m43: m43, m44: m44
    }
}

struct Matrix4<T> {
    m11: T, m12: T, m13: T, m14: T,
    m21: T, m22: T, m23: T, m24: T,
    m31: T, m32: T, m33: T, m34: T,
    m41: T, m42: T, m43: T, m44: T,
}

pub fn main() {}
