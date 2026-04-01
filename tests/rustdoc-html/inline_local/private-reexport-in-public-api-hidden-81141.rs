//@ compile-flags: -Z unstable-options --document-hidden-items

// https://github.com/rust-lang/rust/issues/81141
#![crate_name = "foo"]

#[doc(hidden)]
pub use crate::bar::Bar as Alias;

mod bar {
    pub struct Bar;
}

//@ has 'foo/fn.bar.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar() -> Alias'
pub fn bar() -> Alias {
    Alias
}
