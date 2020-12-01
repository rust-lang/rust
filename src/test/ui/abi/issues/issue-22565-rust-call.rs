#![feature(unboxed_closures)]

extern "rust-call" fn b(_i: i32) {}
//~^ ERROR A function with the "rust-call" ABI must take a single non-self argument that is a tuple

fn main () {
    b(10);
}
