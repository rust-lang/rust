#![crate_name = "foo"]

pub mod sub {
    pub struct Item;

    pub mod prelude {
        pub use super::Item;
    }
}

// @count foo/index.html '//a[@class="mod"][@title="foo::prelude mod"]' 1
pub mod prelude {}

#[doc(inline)]
pub use sub::*;
