//@ check-pass
//@ edition: 2018

mod outer {
    mod inner {
        pub mod inner2 {}
    }
    pub(crate) use inner::{};
    pub(crate) use inner::{{}};
    pub(crate) use inner::{inner2::{}};
    pub(crate) use inner::{inner2::{{}}};
}

fn main() {}
