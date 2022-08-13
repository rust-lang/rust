// compile-flags:--no-defaults --passes strip-priv-imports
// aux-build:empty.rs
// ignore-cross-compile

// @has issue_27104/index.html
// @!hasraw - 'extern crate std'
// @!hasraw - 'use std::prelude::'

// @hasraw - 'pub extern crate empty'
pub extern crate empty;
