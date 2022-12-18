// aux-build: color.rs

extern crate color;

// @is "$.index[*][?(@.inner.name == 'Red')].kind" '"import"'
pub use color::Color::Red;

// @!has "$.index[*][?(@.name == 'Red')]"
// @!has "$.index[*][?(@.name == 'Color')]"
