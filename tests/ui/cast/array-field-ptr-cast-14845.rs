//! Regression test for https://github.com/rust-lang/rust/issues/14845

struct X {
    a: [u8; 1]
}

fn main() {
    let x = X { a: [0] };
    let _f = &x.a as *mut u8; //~ ERROR casting

    let local: [u8; 1] = [0];
    let _v = &local as *mut u8; //~ ERROR casting
}
