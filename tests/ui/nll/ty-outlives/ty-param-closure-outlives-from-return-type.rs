//@ compile-flags:-Zverbose-internals

#![allow(warnings)]
#![feature(rustc_attrs)]

use std::fmt::Debug;

fn with_signature<'a, T, F>(x: Box<T>, op: F) -> Box<dyn Debug + 'a>
    where F: FnOnce(Box<T>) -> Box<dyn Debug + 'a>
{
    op(x)
}

#[rustc_regions]
fn no_region<'a, T>(x: Box<T>) -> Box<dyn Debug + 'a>
where
    T: Debug,
{
    // Here, the closure winds up being required to prove that `T:
    // 'a`.  In principle, it could know that, except that it is
    // type-checked in a fully generic way, and hence it winds up with
    // a propagated requirement that `T: '?2`, where `'?2` appears
    // in the return type. The caller makes the mapping from `'?2` to
    // `'a` (and subsequently reports an error).

    with_signature(x, |y| y)
    //~^ ERROR the parameter type `T` may not live long enough
}

fn correct_region<'a, T>(x: Box<T>) -> Box<Debug + 'a>
where
    T: 'a + Debug,
{
    x
}

fn wrong_region<'a, 'b, T>(x: Box<T>) -> Box<Debug + 'a>
where
    T: 'b + Debug,
{
    x
    //~^ ERROR the parameter type `T` may not live long enough
}

fn outlives_region<'a, 'b, T>(x: Box<T>) -> Box<Debug + 'a>
where
    T: 'b + Debug,
    'b: 'a,
{
    x
}

fn main() {}
