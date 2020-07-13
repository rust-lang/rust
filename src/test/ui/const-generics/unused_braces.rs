// check-pass
// run-rustfix

#![allow(incomplete_features)]
#![warn(unused_braces)]

#![feature(const_generics)]

struct A<const N: usize>;

fn main() {
    let _: A<7>; // ok
    let _: A<{ 7 }>; //~ WARN unnecessary braces
    let _: A<{ 3 + 5 }>; // ok
}
