// xfail-fast
// aux-build:newtype_struct_xc.rs

extern mod newtype_struct_xc;
use newtype_struct_xc::Au;

fn f() -> Au {
    Au(2)
}

fn main() {
    let _ = f();
}

