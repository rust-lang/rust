#![allow(incomplete_features, dead_code)]
#![deny(unconditional_recursion)] //~ note: the lint level is defined here
#![feature(explicit_tail_calls)]

fn f(x: bool) {
    //~^ error: function cannot return without recursing
    //~| note: cannot return without recursing
    if x {
        become f(!x)
    } else {
        f(!x) //~ note: recursive call site
    }
}

// This should *not* lint, tail-recursive functions which never return is a reasonable thing
fn g(x: bool) {
    if x {
        become g(!x)
    } else {
        become g(!x)
    }
}

fn main() {}
