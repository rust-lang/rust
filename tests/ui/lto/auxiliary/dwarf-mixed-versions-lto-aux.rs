//@ compile-flags: -g --crate-type=rlib -Cdwarf-version=4

pub fn say_hi() {
    println!("hello there")
}
