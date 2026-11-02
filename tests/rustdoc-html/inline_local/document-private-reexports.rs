// Test ensures that when we use the `--document-private-items` option, reexports
// work the same as for "public" items.
// Regression test for <https://github.com/rust-lang/rust/issues/159109>.

//@ compile-flags: --document-private-items

#![crate_name = "foo"]

//@ has 'foo/index.html'
// The four reexports, the `not_hidden` and `foofoo` (private) modules.
//@ count - '//*[@class="item-table"]/dt' 6
//@ has - '//*[@class="item-table"]/dt/a[@href="not_hidden/index.html"]' 'not_hidden'
//@ has - '//*[@class="item-table"]/dt/a[@href="struct.Top.html"]' 'Top'
//@ has - '//*[@class="item-table"]/dt/a[@href="struct.NotHidden.html"]' 'NotHidden'
//@ has - '//*[@class="item-table"]/dt/a[@href="struct.Hidden2.html"]' 'Hidden2'
//@ has - '//*[@class="item-table"]/dt/a[@href="struct.X.html"]' 'X'
//@ has - '//*[@class="item-table"]/dt/a[@href="foofoo/index.html"]' 'foofoo'

#[doc(hidden)]
pub(crate) struct TopHidden;

#[doc(inline)]
pub(crate) use TopHidden as Top;

#[doc(hidden)]
mod hidden {
    pub(crate) struct NotHidden;
}

pub(crate) use self::hidden::NotHidden;
// Even though there is no `pub` of any kind, we display it because we're at the
// crate root.
use self::hidden::NotHidden as X;

mod not_hidden {
    #[doc(hidden)]
    pub(crate) struct Hidden;
}

#[doc(inline)]
pub(crate) use self::not_hidden::Hidden as Hidden2;

//@ has 'foo/foofoo/index.html'
//@ count - '//*[@class="item-table"]/dt' 1
//@ has - '//*[@class="item-table"]/dt/a[@href="struct.Something.html"]' 'Something'
mod foofoo {
    // Since there is no `pub` of any kind, we don't display it, even if there is `doc(inline)`.
    #[doc(inline)]
    use crate::not_hidden::Hidden;
    #[doc(inline)]
    pub(crate) use crate::not_hidden::Hidden as Something;
}
