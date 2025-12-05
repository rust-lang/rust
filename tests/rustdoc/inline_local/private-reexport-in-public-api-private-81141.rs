//@ compile-flags: --document-private-items

// https://github.com/rust-lang/rust/issues/81141
#![crate_name = "foo"]

use crate::bar::Bar as Alias;
pub(crate) use crate::bar::Bar as CrateAlias;

mod bar {
    pub struct Bar;
    pub use self::Bar as Inner;
}

// It's a fully private re-export so it should not be displayed.
//@ has 'foo/fn.bar.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar() -> Bar'
pub fn bar() -> Alias {
    Alias
}

// It's public re-export inside a private module so it should be visible.
//@ has 'foo/fn.bar2.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar2() -> Inner'
pub fn bar2() -> crate::bar::Inner {
    Alias
}

// It's a non-public, so it doesn't appear in documentation so it should not be visible.
//@ has 'foo/fn.bar3.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar3() -> Bar'
pub fn bar3() -> CrateAlias {
    Alias
}
