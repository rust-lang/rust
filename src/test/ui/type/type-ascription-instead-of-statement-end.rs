#![feature(type_ascription)]

fn main() {
    println!("test"):
    0; //~ ERROR expected type, found `0`
}

fn foo() {
    println!("test"): 0; //~ ERROR expected type, found `0`
}
