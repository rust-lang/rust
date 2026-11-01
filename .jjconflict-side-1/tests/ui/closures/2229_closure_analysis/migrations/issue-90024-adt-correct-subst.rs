// Test that rustc doesn't ICE as in #90024.
//@ check-pass
// edition=2018

#![warn(rust_2021_incompatible_closure_captures)]

// Checks there's no double-subst into the generic args, otherwise we get OOB
// MCVE by @lqd
pub struct Graph<N, E, Ix> {
    _edges: E,
    _nodes: N,
    _ix: Vec<Ix>,
}
fn graph<N, E>() -> Graph<N, E, i32> {
    todo!()
}
fn first_ice() {
    let g = graph::<i32, i32>();
    let _ = || g;
}

// Checks that there is a subst into the fields, otherwise we get normalization error
// MCVE by @cuviper
use std::iter::Empty;
struct Foo<I: Iterator> {
    data: Vec<I::Item>,
}
pub fn second_ice() {
    let v = Foo::<Empty<()>> { data: vec![] };

    (|| v.data[0])();
}

pub fn main() {
    first_ice();
    second_ice();
}
