#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// Test if we correctly normalize `S<'a>::P` with respect to late-bound regions.

struct S<'a>(&'a ());

trait Inter {
    type P;
}

impl<'a> S<'a> {
    type P = &'a i32;
}

fn ret_ref_local<'e>() -> &'e i32 {
    let f: for<'a> fn(&'a i32) -> S<'a>::P = |x| x;

    let local = 0;
    f(&local) //~ ERROR cannot return value referencing local variable `local`
}

fn main() {}
