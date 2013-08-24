// aux-build:private_variant_xc.rs
// xfail-test

extern mod private_variant_xc;

pub fn main() {
    let _ = private_variant_xc::Bar;
    let _ = private_variant_xc::Baz;    //~ ERROR unresolved name
}
