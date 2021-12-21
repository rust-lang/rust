// aux-build:other_crate.rs

extern crate other_crate;

fn main() {
    let x: Option<i32> = 1i32; //~ ERROR 6:26: 6:30: mismatched types [E0308]
}
