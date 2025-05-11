//@ aux-build: color.rs

//! The purpose of this test it to have a link to [a foreign variant](Red).

extern crate color;
use color::Color::Red;

//@ set red = "$.index[?(@.inner.module.is_crate)].links.Red"

//@ !has "$.index[?(@.name == 'Red')]"
//@ !has "$.index[?(@.name == 'Color')]"
