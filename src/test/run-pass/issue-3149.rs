pure fn Matrix4<T:copy Num>(m11: T, m12: T, m13: T, m14: T,
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

struct Matrix4<T:copy Num> {
    let m11: T; let m12: T; let m13: T; let m14: T;
    let m21: T; let m22: T; let m23: T; let m24: T;
    let m31: T; let m32: T; let m33: T; let m34: T;
    let m41: T; let m42: T; let m43: T; let m44: T;
}

fn main() {}
