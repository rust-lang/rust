//@check-pass
//@aux-build:allow_internal_unstable.rs

extern crate allow_internal_unstable;

allow_internal_unstable::foo!();

fn main(){}
