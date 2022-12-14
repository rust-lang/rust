// aux-build: color.rs

//! The purpose of this test it to have a link to [a foreign variant](Red).

extern crate color;
use color::Color::Red;

// @set red = "$.index[*][?(@.inner.is_crate == true)].links.Red"

// @!has "$.index[*][?(@.name == 'Red')]"
// @!has "$.index[*][?(@.name == 'Color')]"
