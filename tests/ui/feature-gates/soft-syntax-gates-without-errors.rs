//@ check-pass
// This file is used to test the behavior of the early-pass syntax warnings.
// If macro syntax is stabilized, replace with a different unstable syntax.

#[cfg(false)]
macro b() {}
//~^ WARN: `macro` is experimental
//~| WARN: unstable syntax

macro_rules! identity {
    ($($x:tt)*) => ($($x)*);
}

#[cfg(false)]
identity! {
    macro d() {} // No error
}

identity! {
    #[cfg(false)]
    macro e() {}
    //~^ WARN: `macro` is experimental
    //~| WARN: unstable syntax
}

fn main() {}
