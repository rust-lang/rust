//@ aux-build: color.rs

//! The purpose of this test it to have a link to [a foreign variant](Red).

extern crate color;
use color::Color::Red;

//@ jq .index["\(.root)"].links.Red
//@ jq [.index[] | select(.name == "Red" or .name == "Color")] == []
