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

// @matches foo/index.html '//a[@class="test-arrow"][@href="https://www.example.com/?code=%23!%5Ballow(unused)%5D%0Afn%20main()%20%7B%0A%20%20%20%20println!(%22Hello%2C%20world!%22)%3B%0A%7D&edition=2015"]' "Run"
// @matches foo/index.html '//a[@class="test-arrow"][@href="https://www.example.com/?code=%23!%5Ballow(unused)%5D%0Afn%20main()%20%7B%0Aprintln!(%22Hello%2C%20world!%22)%3B%0A%7D&edition=2015"]' "Run"
// @matches foo/index.html '//a[@class="test-arrow"][@href="https://www.example.com/?code=%23!%5Ballow(unused)%5D%0A%23!%5Bfeature(something)%5D%0A%0Afn%20main()%20%7B%0A%20%20%20%20println!(%22Hello%2C%20world!%22)%3B%0A%7D&version=nightly&edition=2015"]' "Run"
