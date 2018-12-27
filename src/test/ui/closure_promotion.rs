// ignore-compare-mode-nll

#![allow(const_err)]

// nll successfully compiles this.
fn main() {
    let x: &'static _ = &|| { let z = 3; z }; //~ ERROR does not live long enough
}
