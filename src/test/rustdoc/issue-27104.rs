// compile-flags:--no-defaults --passes strip-priv-imports
// aux-build:empty.rs
// ignore-cross-compile

// @has issue_27104/index.html
// @!has - 'extern crate std'
// @!has - 'use std::prelude::'

// @has - 'pub extern crate empty'
pub extern crate empty;
