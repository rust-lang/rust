#![feature(fn_delegation)]
#![allow(incomplete_features)]

pub fn external(_: impl FnOnce()) {}

fn delegated_to(_: impl FnOnce()) {}

pub reuse delegated_to as delegated;
