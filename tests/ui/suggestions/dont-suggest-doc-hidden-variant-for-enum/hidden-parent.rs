//@ aux-build:hidden-parent.rs

extern crate hidden_parent;

fn main() {
    let x: Option<i32> = 1i32; //~ ERROR mismatched types
}
