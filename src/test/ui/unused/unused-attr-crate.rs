#![deny(unused_attributes)]

#![feature(naked_functions)]
#![feature(track_caller)]

#![used] //~ ERROR unused attribute
#![non_exhaustive] //~ ERROR unused attribute
#![inline] //~ ERROR unused attribute
#![target_feature(enable = "")] //~ ERROR unused attribute
#![naked] //~ ERROR unused attribute
#![track_caller] //~ ERROR unused attribute

fn main() {}
