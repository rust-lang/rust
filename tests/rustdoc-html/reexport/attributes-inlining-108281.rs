// Regression test for <https://github.com/rust-lang/rust/issues/108281>.
// It ensures that the attributes on the first reexport are not duplicated.

#![crate_name = "foo"]

//@ has 'foo/index.html'

#[doc(hidden)]
pub fn bar() {}
mod sub {
    pub fn public() {}
}

//@ matches - '//dd' '^Displayed$'
/// Displayed
#[doc(inline)]
pub use crate::bar as Bar;
//@ matches - '//dd' '^Hello$'
#[doc(inline)]
/// Hello
pub use crate::Bar as Bar2;

//@ matches - '//dd' '^Public$'
/// Public
pub use crate::sub::public as Public;

//@ has 'foo/fn.Bar.html'
//@ matches - '//div[@class="docblock"]' '^Displayed$'
//@ snapshot fn.Bar.docblock - '//div[@class="docblock"]'

//@ has 'foo/fn.Bar2.html'
//@ matches - '//div[@class="docblock"]' '^Hello|Displayed$'
//@ snapshot fn.Bar2.docblock - '//div[@class="docblock"]'

//@ has 'foo/fn.Public.html'
//@ matches - '//div[@class="docblock"]' '^Public$'
//@ snapshot fn.Public.docblock - '//div[@class="docblock"]'
