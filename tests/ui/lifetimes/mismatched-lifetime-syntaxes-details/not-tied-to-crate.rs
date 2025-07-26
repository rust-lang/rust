#![allow(mismatched_lifetime_syntaxes)]

//! Ensure that the lint level of `mismatched_lifetime_syntaxes` can
//! be adjusted by attributes not applied at the crate-level.

#[warn(mismatched_lifetime_syntaxes)]
mod foo {
    fn bar(x: &'static u8) -> &u8 {
        //~^ WARNING eliding a lifetime that's named elsewhere is confusing
        x
    }

    #[deny(mismatched_lifetime_syntaxes)]
    fn baz(x: &'static u8) -> &u8 {
        //~^ ERROR eliding a lifetime that's named elsewhere is confusing
        x
    }
}

fn main() {}
