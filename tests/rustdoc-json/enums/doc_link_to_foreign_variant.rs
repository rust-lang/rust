//@ aux-build: color.rs

//! The purpose of this test it to have a link to [a foreign variant](Red).

extern crate color;
use color::Color::Red;

//@ jq_is '.index["\(.root)"].links | has("Red")' true
//@ jq_count '[.index[] | select(.name == "Red" or .name == "Color")][]' 0
