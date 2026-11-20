#![allow(non_camel_case_types)]

struct hello(isize);

fn main() {
    let hello = 0; //~ERROR let bindings cannot shadow tuple structs
}
