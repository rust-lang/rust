#![allow(const_err)]

fn main() {
    let _n = 1 / 0; //~ ERROR attempt to divide by zero
}
