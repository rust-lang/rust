#![feature(generic_const_items)]
#![allow(incomplete_features)]

// Ensure that we check if outlives-bounds on const items hold or not.

const C<'a, T: 'a>: () = ();
const K<'a, 'b: 'a>: () = ();

fn parametrized0<'any>() {
    let () = C::<'static, &'any ()>; //~ ERROR lifetime may not live long enough
}

fn parametrized1<'any>() {
    let () = K::<'static, 'any>; //~ ERROR lifetime may not live long enough
}

fn main() {}
