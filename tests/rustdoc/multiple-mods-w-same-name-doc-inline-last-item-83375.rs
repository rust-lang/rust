// https://github.com/rust-lang/rust/issues/83375
#![crate_name = "foo"]

pub mod sub {
    pub struct Item;

    pub mod prelude {
        pub use super::Item;
    }
}

#[doc(inline)]
pub use sub::*;

//@ count foo/index.html '//a[@class="mod"][@title="mod foo::prelude"]' 1
//@ count foo/prelude/index.html '//ul[@class="item-table"]' 0
pub mod prelude {}
