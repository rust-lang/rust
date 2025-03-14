//@ compile-flags: -g --crate-type=rlib -Zdwarf-version=4

pub fn say_hi() {
    println!("hello there")
}
