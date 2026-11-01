#![feature(fn_delegation)]

mod m {}

reuse m::{}; //~ ERROR empty list delegation is not supported

fn main() {}
