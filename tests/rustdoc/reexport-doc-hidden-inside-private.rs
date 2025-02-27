// This test ensures that a re-export of `#[doc(hidden)]` item inside a private
// module will still be displayed (the re-export, not the item).

#![crate_name = "foo"]

mod private_module {
    #[doc(hidden)]
    pub struct Public;
}

//@ has 'foo/index.html'
//@ !has - '//*[@id="reexport.Foo"]/code' 'pub use crate::private_module::Public as Foo;'
pub use crate::private_module::Public as Foo;
// Glob re-exports with no visible items should not be displayed.
//@ count - '//*[@class="item-table reexports"]/dt' 0
pub use crate::private_module::*;
