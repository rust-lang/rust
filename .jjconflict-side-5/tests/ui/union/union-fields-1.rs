#![deny(dead_code)]

union U1 {
    a: u8, // should not be reported
    b: u8, // should not be reported
    c: u8, //~ ERROR field `c` is never read
}
union U2 {
    a: u8, //~ ERROR field `a` is never read
    b: u8, // should not be reported
    c: u8, // should not be reported
}
union NoDropLike { a: u8 } //~ ERROR field `a` is never read

union U {
    a: u8, // should not be reported
    b: u8, // should not be reported
    c: u8, //~ ERROR field `c` is never read
}
type A = U;

fn main() {
    let u = U1 { a: 0 };
    let _a = unsafe { u.b };

    let u = U2 { c: 0 };
    let _b = unsafe { u.b };

    let _u = NoDropLike { a: 10 };
    let u = A { a: 0 };
    let _b = unsafe { u.b };
}
