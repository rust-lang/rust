// check-pass

#![feature(dyn_star)]
//~^ WARN the feature `dyn_star` is incomplete and may not be safe to use and/or cause compiler crashes

fn _foo() -> dyn* Unpin {
    4usize
}

fn main() {}
