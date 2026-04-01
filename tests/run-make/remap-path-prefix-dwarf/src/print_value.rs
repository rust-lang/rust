#![crate_type = "rlib"]

extern crate some_value;

pub fn print_value() {
    println!("{}", some_value::get_some_value());
}
