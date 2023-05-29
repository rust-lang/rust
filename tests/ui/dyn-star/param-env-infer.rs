// revisions: current next
//[next] compile-flags: -Ztrait-solver=next
// incremental

#![feature(dyn_star, pointer_like_trait)]
//~^ WARN the feature `dyn_star` is incomplete and may not be safe to use and/or cause compiler crashes

use std::fmt::Debug;
use std::marker::PointerLike;

fn make_dyn_star<'a, T: PointerLike + Debug + 'a>(t: T) -> impl PointerLike + Debug + 'a {
    //[next]~^ ERROR cycle detected when computing type of `make_dyn_star::{opaque#0}`
    t as _
    //[current]~^ ERROR type annotations needed
}

fn main() {}
