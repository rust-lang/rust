//@ check-fail
// This file is used to test the behavior of the early-pass syntax warnings.
// If macro syntax is stabilized, replace with a different unstable syntax.

macro a() {}
//~^ ERROR: `macro` is experimental

#[cfg(FALSE)]
macro b() {}

macro_rules! identity {
    ($($x:tt)*) => ($($x)*);
}

identity! {
    macro c() {}
    //~^ ERROR: `macro` is experimental
}

#[cfg(FALSE)]
identity! {
    macro d() {} // No error
}

identity! {
    #[cfg(FALSE)]
    macro e() {}
}

fn main() {}
