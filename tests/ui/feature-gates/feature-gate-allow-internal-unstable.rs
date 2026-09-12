#![allow(unused_macros)]

#[allow_internal_unstable()] //~ ERROR the `allow_internal_unstable` attribute side-steps
macro_rules! foo {
    () => {}
}

fn main() {}
