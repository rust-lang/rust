// Regression test for <https://github.com/rust-lang/rust/issues/108281>.
// It ensures that the attributes on the first reexport are not duplicated.

#![crate_name = "foo"]

// @has 'foo/index.html'

#[doc(hidden)]
pub fn bar() {}
mod sub {
    pub fn public() {}
}

// @matches - '//*[@class="desc docblock-short"]' '^Displayed$'
/// Displayed
#[doc(inline)]
pub use crate::bar as Bar;
// @matches - '//*[@class="desc docblock-short"]' '^Hello\sDisplayed$'
#[doc(inline)]
/// Hello
pub use crate::Bar as Bar2;

// @matches - '//*[@class="desc docblock-short"]' '^Public$'
/// Public
pub use crate::sub::public as Public;
