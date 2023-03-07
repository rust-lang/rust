// run-pass

#![feature(dyn_star)]
//~^ WARN the feature `dyn_star` is incomplete and may not be safe to use and/or cause compiler crashes

use std::fmt::Debug;

fn main() {
    let x: dyn* Debug = &42;
    let x = Box::new(x) as Box<dyn Debug>;
    assert_eq!("42", format!("{x:?}"));

    // Also test opposite direction.
    let x: Box<dyn Debug> = Box::new(42);
    let x = &x as dyn* Debug;
    assert_eq!("42", format!("{x:?}"));
}
