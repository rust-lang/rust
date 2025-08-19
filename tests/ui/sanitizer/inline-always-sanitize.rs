//@ check-pass

#![feature(sanitize)]

#[inline(always)]
//~^ NOTE inlining requested here
#[sanitize(address = "off")]
//~^ WARN  setting `sanitize` off will have no effect after inlining
//~| NOTE on by default
fn x() {
}

fn main() {
    x()
}
