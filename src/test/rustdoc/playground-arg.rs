// compile-flags: --playground-url=https://example.com/ -Z unstable-options
// ignore-tidy-linelength

#![crate_name = "foo"]

//! ```
//! use foo::dummy;
//! dummy();
//! ```

pub fn dummy() {}

// ensure that `extern crate foo;` was inserted into code snips automatically:
// @matches foo/index.html '//a[@class="test-arrow"][@href="https://example.com/?code=%23!%5Ballow(unused)%5D%0Aextern%20crate%20foo%3B%0Afn%20main()%20%7B%0Ause%20foo%3A%3Adummy%3B%0Adummy()%3B%0A%7D&edition=2015"]' "Run"
