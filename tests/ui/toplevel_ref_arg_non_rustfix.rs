#![warn(clippy::toplevel_ref_arg)]
#![allow(unused)]

fn the_answer(ref mut x: u8) {
    *x = 42;
}

fn main() {
    let mut x = 0;
    the_answer(x);
}
