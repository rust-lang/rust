//@ aux-build: color.rs

extern crate color;

// @has "$.index[*].inner.import[?(@.name == 'Red')]"
pub use color::Color::Red;

// @!has "$.index[*][?(@.name == 'Red')]"
// @!has "$.index[*][?(@.name == 'Color')]"
