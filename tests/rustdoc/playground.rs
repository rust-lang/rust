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

// @matches foo/index.html '//a[@class="test-arrow"][@href="https://www.example.com/?code=%23![allow(unused)]%0Afn+main()+{%0Aprintln!(%22Hello,+world!%22);%0A}&edition=2015"]' "Run"
// @matches foo/index.html '//a[@class="test-arrow"][@href="https://www.example.com/?code=%23![allow(unused)]%0Afn+main()+{%0A++++println!(%22Hello,+world!%22);%0A}&edition=2015"]' "Run"
// @matches foo/index.html '//a[@class="test-arrow"][@href="https://www.example.com/?code=%23![allow(unused)]%0A%23![feature(something)]%0A%0Afn+main()+{%0A++++println!(%22Hello,+world!%22);%0A}&version=nightly&edition=2015"]' "Run"
