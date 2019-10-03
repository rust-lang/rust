#![feature(plugin)]
#![plugin(foo="bleh")] //~ ERROR malformed `plugin` attribute
//~| WARN use of deprecated attribute `plugin`: compiler plugins are deprecated

fn main() {}
