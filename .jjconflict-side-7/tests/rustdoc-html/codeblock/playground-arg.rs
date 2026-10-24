//@ compile-flags: --playground-url=https://example.com/ -Z unstable-options

#![crate_name = "foo"]

//! ```
//! use foo::dummy;
//! dummy();
//! ```

pub fn dummy() {}

// ensure that `extern crate foo;` was inserted into code snips automatically:
//@ matches foo/index.html '//a[@class="test-arrow"][@href="https://example.com/?code=%23!%5Ballow(unused)%5D%0A%23%5Ballow(unused_extern_crates)%5D%0Aextern+crate+r%23foo;%0Afn+main()+%7B%0A++++use+foo::dummy;%0A++++dummy();%0A%7D&edition=2015"]' ""
