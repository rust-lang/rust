#![feature(plugin)]
#![plugin] //~ ERROR malformed `plugin` attribute
//~| WARN use of deprecated attribute `plugin`: compiler plugins are deprecated

fn main() {}
