#![feature(dyn_star)]
#![allow(incomplete_features)]

use std::fmt::Display;

pub fn require_dyn_star_display(_: dyn* Display) {}

fn works_locally() {
    require_dyn_star_display(1usize);
}
