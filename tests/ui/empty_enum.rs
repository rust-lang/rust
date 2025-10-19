#![warn(clippy::empty_enum)]
// Enable never type to test empty enum lint
#![feature(never_type)]

enum Empty {}
//~^ empty_enum

mod issue15910 {
    enum NotReallyEmpty {
        #[cfg(false)]
        Hidden,
    }

    enum OneVisibleVariant {
        #[cfg(false)]
        Hidden,
        Visible,
    }

    enum CfgInsideVariant {
        Variant(#[cfg(false)] String),
    }
}

fn main() {}
