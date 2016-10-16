// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

#![crate_name = "foo"]

#![doc(html_playground_url = "https://www.example.com/")]

//! module docs
//!
//! ```
//! println!("Hello, world!");
//! ```
//!
//! ```
//! fn main() {
//!     println!("Hello, world!");
//! }
//! ```
//!
//! ```
//! #![feature(something)]
//!
//! fn main() {
//!     println!("Hello, world!");
//! }
//! ```

// @matches foo/index.html '//a[@class="test-arrow"][@href="https://www.example.com/?code=fn%20main()%20%7B%0A%20%20%20%20println!(%22Hello%2C%20world!%22)%3B%0A%7D%0A"]' "Run"
// @matches foo/index.html '//a[@class="test-arrow"][@href="https://www.example.com/?code=fn%20main()%20%7B%0Aprintln!(%22Hello%2C%20world!%22)%3B%0A%7D"]' "Run"
// @matches foo/index.html '//a[@class="test-arrow"][@href="https://www.example.com/?code=%23!%5Bfeature(something)%5D%0A%0Afn%20main()%20%7B%0A%20%20%20%20println!(%22Hello%2C%20world!%22)%3B%0A%7D%0A&version=nightly"]' "Run"
