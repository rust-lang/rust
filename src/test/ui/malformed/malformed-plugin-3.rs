// ignore-tidy-linelength

#![feature(plugin)]
#![plugin(foo="bleh")] //~ ERROR malformed `plugin` attribute
//~| WARN use of deprecated attribute `plugin`: compiler plugins are deprecated and will be removed in 1.44.0

fn main() {}
