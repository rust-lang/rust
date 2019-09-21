// ignore-tidy-linelength

#![deny(unused_attributes)]
#![feature(plugin)]

#[plugin(bla)]  //~ ERROR unused attribute
                //~^ ERROR should be an inner attribute
//~| WARN use of deprecated attribute `plugin`: compiler plugins are deprecated and will be removed in 1.44.0

fn main() {}
