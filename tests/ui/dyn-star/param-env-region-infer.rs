// revisions: current next
//[next] compile-flags: -Ztrait-solver=next
//[next] check-pass
// incremental

// checks that we don't ICE if there are region inference variables in the environment
// when computing `PointerLike` builtin candidates.

#![feature(dyn_star, pointer_like_trait)]
#![allow(incomplete_features)]

use std::fmt::Debug;
use std::marker::PointerLike;

fn make_dyn_star<'a, T: PointerLike + Debug + 'a>(t: T) -> impl PointerLike + Debug + 'a {
    t as _
    //[current]~^ ERROR type annotations needed
}

fn main() {}
