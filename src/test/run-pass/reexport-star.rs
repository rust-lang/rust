mod a {
    pub fn f() {}
    pub fn g() {}
}

mod b {
    pub use a::*;
}

fn main() {
    b::f();
    b::g();
}

