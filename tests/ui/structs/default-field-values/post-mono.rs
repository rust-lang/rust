//@ build-fail
//@ revisions: direct indirect

#![feature(default_field_values)]

struct Z<const X: usize> {
    post_mono: usize = X / 0,
    //~^ ERROR evaluation of `Z::<1>::post_mono::{constant#0}` failed
}

fn indirect<const X: usize>() {
    let x: Z<X> = Z { .. };
}

#[cfg(direct)]
fn main() {
    let x: Z<1> = Z { .. };
}

#[cfg(indirect)]
fn main() {
    indirect::<1>();
}
