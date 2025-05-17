//@ aux-build: issue-113982-doc_auto_cfg-reexport-foreign.rs

// https://github.com/rust-lang/rust/issues/113982
#![feature(no_core, doc_cfg)]
#![no_core]
#![crate_name = "foo"]

extern crate colors;

//@ has 'foo/index.html' '//*[@class="stab portability"]' 'Non-colors'
//@ has 'foo/struct.Color.html' '//*[@class="stab portability"]' \
//      'Available on non-crate feature colors only.'
#[cfg(not(feature = "colors"))]
pub use colors::*;

//@ has 'foo/index.html' '//*[@class="stab portability"]' 'Non-fruits'
//@ has 'foo/struct.Red.html' '//*[@class="stab portability"]' \
//      'Available on non-crate feature fruits only.'
#[cfg(not(feature = "fruits"))]
pub use colors::Color as Red;
