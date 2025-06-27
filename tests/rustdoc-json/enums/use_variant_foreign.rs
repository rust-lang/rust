//@ aux-build: color.rs

extern crate color;

//@ jq .index[] | select(.inner.use.name? == "Red")
pub use color::Color::Red;

//@ jq [.index[] | select(.name == "Red" or .name == "Color")] == []
