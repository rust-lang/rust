#![feature(plugin)]
#![plugin(foo="bleh")] //~ ERROR malformed `plugin` attribute

fn main() {}
