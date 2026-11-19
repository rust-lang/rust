//! Regression test for <https://github.com/rust-lang/rust/issues/150749>

#![feature(
    gca_min_const_items,
    gca_macroless_args,
)]

trait Trait {
    fn not_a_struct();
}

impl Trait for () {
    fn not_a_struct() {
        let _: [(); Self];
        //~^ ERROR the `Self` constructor cannot be used as a const argument
    }
}


fn main() {}
