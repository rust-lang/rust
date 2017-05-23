#![feature(plugin)]
#![plugin(clippy)]

#![warn(let_unit_value)]
#![allow(unused_variables)]

macro_rules! let_and_return {
    ($n:expr) => {{
        let ret = $n;
    }}
}

fn main() {
    let _x = println!("x");
    let _y = 1;   // this is fine
    let _z = ((), 1);  // this as well
    if true {
        let _a = ();
    }

    let_and_return!(()) // should be fine
}

#[derive(Copy, Clone)]
pub struct ContainsUnit(()); // should be fine
