//! Regression test for <https://github.com/rust-lang/rust/issues/17999>.

#![deny(unused_variables)]

fn main() {
    for _ in 1..101 {
        let x = (); //~ ERROR: unused variable: `x`
        match () {
            a => {} //~ ERROR: unused variable: `a`
        }
    }
}
