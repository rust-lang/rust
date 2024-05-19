// Regression test for <https://github.com/rust-lang/rust/issues/108231>.
// Macros with `#[macro_export]` attribute should be visible at the top level
// even if they are inside a doc hidden item.

#![crate_name = "foo"]

// @has 'foo/index.html'
// @count - '//*[@id="main-content"]//a[@class="macro"]' 1
// @has - '//*[@id="main-content"]//a[@class="macro"]' 'foo'

#[doc(hidden)]
pub mod __internal {
    /// This one should be visible.
    #[macro_export]
    macro_rules! foo {
        () => {};
    }

    /// This one should be hidden.
    macro_rules! bar {
        () => {};
    }
}
