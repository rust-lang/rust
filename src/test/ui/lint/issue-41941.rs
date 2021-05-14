// Checks whether exceptions can be added to #![deny(warnings)].
// check-pass

#![deny(warnings)]
#![warn(unused_variables)]

fn main() {
    let a = 5;
    //~^ WARNING: unused
}
