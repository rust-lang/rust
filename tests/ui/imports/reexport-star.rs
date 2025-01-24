//@ run-pass

mod a {
    pub fn f() {}
    pub fn g() {}
}

mod b {
    pub use a::*;
}

pub fn main() {
    b::f();
    b::g();
}
