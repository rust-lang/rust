// run-pass
// aux-build:struct_variant_xc_aux.rs

extern crate struct_variant_xc_aux;

use struct_variant_xc_aux::Enum::{StructVariant, Variant};

pub fn main() {
    let arg = match (StructVariant { arg: 42 }) {
        Variant(_) => unreachable!(),
        StructVariant { arg } => arg
    };
    assert_eq!(arg, 42);
}
