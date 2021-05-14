// Checks that `-A warnings` on the command line can be overridden.
// compile-flags: -A warnings
// check-pass

#![warn(unused)]

fn main() {
    let a = 5;
    //~^ WARNING: unused
}
