//@ check-pass
//@ compile-flags: -Znext-solver

use std::fmt::Display;

fn box_dyn_display(_: Box<dyn Display>) {}

fn main() {
    // During coercion, we don't necessarily know whether `{integer}` implements
    // `Display`. Before, that would cause us to bail out in the coercion loop when
    // checking `{integer}: Unsize<dyn Display>`.
    box_dyn_display(Box::new(1));
}
