//@ build-fail
//@ revisions: direct indirect

#![feature(default_field_values)]

struct Z<const X: usize> {
    post_mono: usize = X / 0,
    //~^ ERROR attempt to divide `1_usize` by zero
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
