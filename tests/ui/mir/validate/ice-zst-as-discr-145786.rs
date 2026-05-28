// Do not attempt to take the discriminant as the source
// converted to a `u128`, that won't work for ZST.
//
//@ compile-flags: -Zvalidate-mir

enum A {
    B,
    C,
}

fn main() {
    let _: A = unsafe { std::mem::transmute(()) };
    //~^ ERROR cannot transmute between types of different sizes, or dependently-sized types
}
