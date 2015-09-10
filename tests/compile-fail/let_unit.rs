#![feature(plugin)]
#![plugin(clippy)]

#![deny(let_unit_value)]
#![allow(unused_variables)]

macro_rules! let_and_return {
    ($n:expr) => {{
        let ret = $n;
    }}
}

fn main() {
    let _x = println!("x");  //~ERROR this let-binding has unit value
    let _y = 1;   // this is fine
    let _z = ((), 1);  // this as well
    if true {
        let _a = ();  //~ERROR this let-binding has unit value
    }

    let_and_return!(()) // should be fine
}

#[derive(Copy, Clone)]
pub struct ContainsUnit(()); // should be fine
