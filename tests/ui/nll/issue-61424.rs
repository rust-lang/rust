//@ run-rustfix

#![deny(unused_mut)]

fn main() {
    let mut x; //~ ERROR: variable does not need to be mutable
    x = String::new();
    dbg!(x);
}
