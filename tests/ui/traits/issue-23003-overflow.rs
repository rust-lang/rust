// A variant of traits-issue-23003 in which an infinite series of
// types are required. This test now just compiles fine, since the
// relevant rules that triggered the overflow were removed.

//@ check-pass
#![allow(dead_code)]

use std::marker::PhantomData;

trait Async {
    type Cancel;
}

struct Receipt<A:Async> {
    marker: PhantomData<A>,
}

struct Complete<B> {
    core: Option<B>,
}

impl<B> Async for Complete<B> {
    type Cancel = Receipt<Complete<Option<B>>>;
}

fn foo(_: Receipt<Complete<()>>) { }


fn main() { }
