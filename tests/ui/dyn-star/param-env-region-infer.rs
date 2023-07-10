// revisions: current next
// Need `-Zdeduplicate-diagnostics=yes` because the number of cycle errors
// emitted is for some horrible reason platform-specific.
//[next] compile-flags: -Ztrait-solver=next -Zdeduplicate-diagnostics=yes
// incremental

// checks that we don't ICE if there are region inference variables in the environment
// when computing `PointerLike` builtin candidates.

#![feature(dyn_star, pointer_like_trait)]
#![allow(incomplete_features)]

use std::fmt::Debug;
use std::marker::PointerLike;

fn make_dyn_star<'a, T: PointerLike + Debug + 'a>(t: T) -> impl PointerLike + Debug + 'a {
    //[next]~^ ERROR cycle detected when computing
    t as _
    //[current]~^ ERROR type annotations needed
}

fn main() {}
