// compile-flags: -Zunstable-options --diagnostic-width=10
#![deny(rustdoc::bare_urls)]

/// This is a long line that contains a http://link.com
pub struct Foo; //~^ ERROR
