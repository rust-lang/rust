// aux-build:use_crate.rs
// aux-build:use_crate_2.rs
// build-aux-docs
// edition:2018
// compile-flags:--extern use_crate --extern use_crate_2 -Z unstable-options

// During the buildup to Rust 2018, rustdoc would eagerly inline `pub use some_crate;` as if it
// were a module, so we changed it to make `pub use`ing crate roots remain as a `pub use` statement
// in docs... unless you added `#[doc(inline)]`.

#![crate_name = "local"]

// @!has-dir local/use_crate
// @has local/index.html
// @has - '//code' 'pub use use_crate'
pub use use_crate;

// @has-dir local/asdf
// @has local/asdf/index.html
// @has local/index.html '//a/@href' 'asdf/index.html'
pub use use_crate::asdf;

// @has-dir local/use_crate_2
// @has local/use_crate_2/index.html
// @has local/index.html '//a/@href' 'use_crate_2/index.html'
#[doc(inline)]
pub use use_crate_2;
