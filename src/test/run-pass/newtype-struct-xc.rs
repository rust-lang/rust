// xfail-fast
// aux-build:newtype_struct_xc.rs

extern mod newtype_struct_xc;

pub fn main() {
    let _ = newtype_struct_xc::Au(2);
}
