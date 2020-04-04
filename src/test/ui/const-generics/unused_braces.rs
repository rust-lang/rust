// check-pass
#![warn(unused_braces)]

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct A<const N: usize>;

fn main() {
    let _: A<7>; // ok
    let _: A<{ 7 }>; //~ WARN unnecessary braces
    let _: A<{ 3 + 5 }>; // ok
}
