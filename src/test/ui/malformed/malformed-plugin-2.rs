#![feature(plugin)]
#![plugin="bleh"] //~ ERROR malformed `plugin` attribute
//~| WARN use of deprecated attribute `plugin`: compiler plugins are deprecated

fn main() {}
