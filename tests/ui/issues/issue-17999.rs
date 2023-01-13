#![deny(unused_variables)]

fn main() {
    for _ in 1..101 {
        let x = (); //~ ERROR: unused variable: `x`
        match () {
            a => {} //~ ERROR: unused variable: `a`
        }
    }
}
