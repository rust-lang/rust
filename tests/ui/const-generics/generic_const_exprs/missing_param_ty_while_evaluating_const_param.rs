#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params)]
struct X<
    const FN: () = {
        || {
            let _: [(); B]; //~ ERROR cannot find value `B` in this scope
            //~^ ERROR the constant `FN` is not of type `()`
        };
    },
>;
fn main() {}
