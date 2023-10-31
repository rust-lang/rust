// compile-flags: -Z unstable-options --document-hidden-items

#![crate_name = "foo"]

#[doc(hidden)]
pub use crate::bar::Bar as Alias;

mod bar {
    pub struct Bar;
}

// @has 'foo/fn.bar.html'
// @has - '//*[@class="rust item-decl"]/code' 'pub fn bar() -> Alias'
pub fn bar() -> Alias {
    Alias
}
