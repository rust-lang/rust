// run-pass
// aux-build:struct_variant_xc_aux.rs
// pretty-expanded FIXME #23616

extern crate struct_variant_xc_aux;

use struct_variant_xc_aux::Enum::StructVariant;

pub fn main() {
    let _ = StructVariant { arg: 1 };
}
