#![deny(unused_attributes)]
#![feature(plugin)]

#[plugin(bla)]  //~ ERROR unused attribute
                //~^ ERROR should be an inner attribute

fn main() {}
