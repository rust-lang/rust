#![crate_name = "foo"]

pub mod sub {
    pub struct Item;

    pub mod prelude {
        pub use super::Item;
    }
}

#[doc(inline)]
pub use sub::*;

// @count foo/index.html '//a[@class="mod"][@title="foo::prelude mod"]' 1
// @count foo/prelude/index.html '//div[@class="item-row"]' 0
pub mod prelude {}
