#![deny(rustdoc::unportable_markdown)]
// invalid_html is buggy in this case
#![allow(rustdoc::invalid_html_tags)]

/// [doc.example]: https://example.com
///
/// - bar
//~^ ERROR
//~| ERROR
#[doc(inline)]
pub use foo::First;

pub mod foo {
/// - My [doc.example]
//~^ ERROR
pub struct First;
}
