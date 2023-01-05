// run-pass
// Test stack overflow triggered by evaluating the implications. To be
// WF, the type `Receipt<Complete>` would require that `<Complete as
// Async>::Cancel` be WF. This normalizes to `Receipt<Complete>`
// again, leading to an infinite cycle. Issue #23003.

// pretty-expanded FIXME #23616

#![allow(dead_code)]
#![allow(unused_variables)]

use std::marker::PhantomData;

trait Async {
    type Cancel;
}

struct Receipt<A:Async> {
    marker: PhantomData<A>,
}

struct Complete {
    core: Option<()>,
}

impl Async for Complete {
    type Cancel = Receipt<Complete>;
}

fn foo(r: Receipt<Complete>) { }

fn main() { }
