#![crate_name = "foo"]

#![doc(html_playground_url = "")]

//! module docs
//!
//! ```
//! println!("Hello, world!");
//! ```

// @!has foo/index.html '//a[@class="test-arrow"]' "Run"
