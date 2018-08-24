// aux-build:variants.rs
extern crate variants;

use variants::NonExhaustiveVariants;

/*
 * The initial implementation of #[non_exhaustive] (RFC 2008) does not include support for
 * variants. See issue #44109 and PR 45394.
 */
// ignore-test

fn main() {
    let variant_struct = NonExhaustiveVariants::Struct { field: 640 };
    //~^ ERROR cannot create non-exhaustive variant

    let variant_tuple = NonExhaustiveVariants::Tuple { 0: 640 };
    //~^ ERROR cannot create non-exhaustive variant

    match variant_struct {
        NonExhaustiveVariants::Unit => "",
        NonExhaustiveVariants::Tuple(fe_tpl) => "",
        //~^ ERROR `..` required with variant marked as non-exhaustive
        NonExhaustiveVariants::Struct { field } => ""
        //~^ ERROR `..` required with variant marked as non-exhaustive
    };
}
