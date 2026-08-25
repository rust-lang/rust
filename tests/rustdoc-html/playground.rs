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

//@ matches foo/index.html '//a[@class="test-arrow"][@href="https://www.example.com/?code=%23!%5Ballow(unused)%5D%0Afn+main()+%7B%0A++++println!(%22Hello,+world!%22);%0A%7D&edition=2015"]' ""
//@ matches foo/index.html '//a[@class="test-arrow"][@href="https://www.example.com/?code=%23!%5Ballow(unused)%5D%0Afn+main()+%7B%0A++++println!(%22Hello,+world!%22);%0A%7D&edition=2015"]' ""
//@ matches foo/index.html '//a[@class="test-arrow"][@href="https://www.example.com/?code=%23!%5Ballow(unused)%5D%0A%23!%5Bfeature(something)%5D%0A%0A%0Afn+main()+%7B%0A++++println!(%22Hello,+world!%22);%0A%7D&version=nightly&edition=2015"]' ""
