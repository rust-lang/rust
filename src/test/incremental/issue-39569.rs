// Regression test for a weird corner case in our dep-graph reduction
// code. When we solve `CoerceUnsized<Foo>`, we find no impls, so we
// don't end up with an edge to any HIR nodes, but it still gets
// preserved in the dep graph.

// revisions:rpass1 rpass2
// compile-flags: -Z query-dep-graph

use std::sync::Arc;

#[cfg(rpass1)]
struct Foo { x: usize }

#[cfg(rpass1)]
fn main() {
    let x: Arc<Foo> = Arc::new(Foo { x: 22 });
    let y: Arc<Foo> = x;
}

#[cfg(rpass2)]
struct FooX { x: usize }

#[cfg(rpass2)]
fn main() {
    let x: Arc<FooX> = Arc::new(FooX { x: 22 });
    let y: Arc<FooX> = x;
}
