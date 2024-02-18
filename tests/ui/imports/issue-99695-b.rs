//@ run-rustfix
#![allow(unused, nonstandard_style)]
mod m {

    mod p {
        #[macro_export]
        macro_rules! nu {
            {} => {};
        }

        pub struct other_item;
    }

    pub use self::p::{nu, other_item as _};
    //~^ ERROR unresolved import `self::p::nu` [E0432]
    //~| HELP a macro with this name exists at the root of the crate
}

fn main() {}
