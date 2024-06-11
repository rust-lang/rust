//@ known-bug: rust-lang/rust#125992
//@ compile-flags:  -Zpolonius=next

#![feature(inherent_associated_types)]

type Function = for<'a> fn(&'a i32) -> S<'a>::P;

struct S<'a>(&'a ());

impl<'a> S {
    type P = &'a i32;
}

fn ret_ref_local<'e>() -> &'e i32 {
    let f: Function = |x| x;

    let local = 0;
    f(&local)
}
