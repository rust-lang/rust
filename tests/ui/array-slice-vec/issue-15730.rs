//@ run-pass
#![allow(unused_mut)]
#![allow(unused_variables)]

fn main() {
    let mut array = [1, 2, 3];
    let pie_slice = &array[1..2];
}
