// Ensure that inlined reexport of hidden macros is working as expected.
// Part of <https://github.com/rust-lang/rust/issues/59368>.

#![crate_name = "foo"]

// @has 'foo/index.html'
// @has - '//*[@id="main-content"]//a[@href="macro.Macro2.html"]' 'Macro2'

// @has 'foo/macro.Macro2.html'
// @has - '//*[@class="docblock"]' 'Displayed'

#[macro_export]
#[doc(hidden)]
macro_rules! foo {
    () => {};
}

/// not displayed
pub use crate::foo as Macro;
/// Displayed
#[doc(inline)]
pub use crate::foo as Macro2;
