// Test ensures that when we use the `--document-private-items` option, reexports
// work the same as for "public" items.
// Regression test for <https://github.com/rust-lang/rust/issues/159109>.

//@ compile-flags: --document-private-items

#![crate_name = "foo"]

//@ has 'foo/index.html'
// The three reexports and the `not_hidden` (private) module.
//@ count - '//*[@class="item-table"]/dt' 4
//@ has - '//*[@class="item-table"]/dt/a[@href="not_hidden/index.html"]' 'not_hidden'
//@ has - '//*[@class="item-table"]/dt/a[@href="struct.Top.html"]' 'Top'
//@ has - '//*[@class="item-table"]/dt/a[@href="struct.NotHidden.html"]' 'NotHidden'
//@ has - '//*[@class="item-table"]/dt/a[@href="struct.Hidden2.html"]' 'Hidden2'

#[doc(hidden)]
pub(crate) struct TopHidden;

#[doc(inline)]
pub(crate) use TopHidden as Top;

#[doc(hidden)]
mod hidden {
    pub(crate) struct NotHidden;
}

pub(crate) use self::hidden::NotHidden;
// Since there is no `pub` of any kind, we don't display it.
use self::hidden::NotHidden as X;

mod not_hidden {
    #[doc(hidden)]
    pub(crate) struct Hidden;
}

#[doc(inline)]
pub(crate) use self::not_hidden::Hidden as Hidden2;
// Since there is no `pub` of any kind, we don't display it, even if there is `doc(inline)`.
#[doc(inline)]
use self::not_hidden::Hidden;
