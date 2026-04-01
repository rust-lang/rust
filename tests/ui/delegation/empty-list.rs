#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod m {}

reuse m::{}; //~ ERROR empty list delegation is not supported

fn main() {}
