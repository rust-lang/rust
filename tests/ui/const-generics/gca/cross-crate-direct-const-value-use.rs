//@ run-pass
//@ aux-crate:up=direct_const_item.rs

const N: usize = up::N;

fn main() {
    println!("{N}");
}
