#![crate_name = "b"]

extern crate a;

fn main() {
    a::foo(22_u32);
}
