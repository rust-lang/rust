#![allow(unused_macros)]

#[allow_internal_unstable()] //~ ERROR allow_internal_unstable side-steps
macro_rules! foo {
    () => {}
}

fn main() {}
