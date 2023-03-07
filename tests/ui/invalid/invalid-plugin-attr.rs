#![deny(unused_attributes)]
#![feature(plugin)]

#[plugin(bla)] //~ ERROR should be an inner attribute
//~| WARN use of deprecated attribute `plugin`: compiler plugins are deprecated

fn main() {}
