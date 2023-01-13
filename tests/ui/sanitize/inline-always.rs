// check-pass

#![feature(no_sanitize)]

#[inline(always)]
//~^ NOTE inlining requested here
#[no_sanitize(address)]
//~^ WARN will have no effect after inlining
//~| NOTE on by default
fn x() {
}

fn main() {
    x()
}
