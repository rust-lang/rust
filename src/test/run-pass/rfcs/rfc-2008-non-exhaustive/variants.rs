// run-pass
// aux-build:variants.rs
extern crate variants;

use variants::NonExhaustiveVariants;

/*
 * The initial implementation of #[non_exhaustive] (RFC 2008) does not include support for
 * variants. See issue #44109 and PR 45394.
 */
// ignore-test

fn main() {
    let variant_tuple = NonExhaustiveVariants::Tuple { 0: 340 };
    let variant_struct = NonExhaustiveVariants::Struct { field: 340 };

    match variant_struct {
        NonExhaustiveVariants::Unit => "",
        NonExhaustiveVariants::Struct { field, .. } => "",
        NonExhaustiveVariants::Tuple(fe_tpl, ..) => ""
    };
}
