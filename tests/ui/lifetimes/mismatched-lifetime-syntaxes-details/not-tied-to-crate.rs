#![allow(mismatched_lifetime_syntaxes)]

//! Ensure that the lint level of `mismatched_lifetime_syntaxes` can
//! be adjusted by attributes not applied at the crate-level.

#[warn(mismatched_lifetime_syntaxes)]
mod foo {
    fn bar(x: &'static u8) -> &u8 {
        //~^ WARNING lifetime flowing from input to output with different syntax
        x
    }

    #[deny(mismatched_lifetime_syntaxes)]
    fn baz(x: &'static u8) -> &u8 {
        //~^ ERROR lifetime flowing from input to output with different syntax
        x
    }
}

fn main() {}
