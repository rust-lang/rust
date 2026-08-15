//@ edition: 2024
//@ revisions: no_dxf dxf
//@[no_dxf] compile-flags: -Znext-solver
//@[no_dxf] check-pass
//@[dxf] compile-flags: -Znext-solver -Zdxf
//@[dxf] check-pass

#![allow(unused)]

use std::future::Future;

fn assert_send<T: Send>(_: T) {}

// === Case 1: dyn for<'a> Fn held across await creates U1 placeholder ===
// The `for<'a>` in `Box<dyn for<'a> Fn(&'a i32) + Send>` creates
// PlaceholderRegion(!1) when the solver decomposes the witness for Send.
async fn test_dyn_fn() {
    let f: Box<dyn for<'a> Fn(&'a i32) + Send> = Box::new(|_| {});
    async {}.await;
    f(&1);
}

// === Case 2: Conditional Send via for<'a> bound creates U1 placeholder ===
trait Foo<'a> {}
impl<'a> Foo<'a> for () {}

struct Bar<T>(T);
unsafe impl<T> Send for Bar<T> where T: for<'a> Foo<'a> {}

async fn test_hrtb() {
    let x = Bar(());
    async {}.await;
    drop(x);
}

// === Case 3: GAT with for<'a> Send bound creates U1 placeholder ===
trait GatTrait {
    type Assoc<'a>;
}

impl GatTrait for () {
    type Assoc<'a> = &'a i32;
}

struct GatBar<T: GatTrait>(T);
unsafe impl<T: GatTrait> Send for GatBar<T>
where
    for<'a> <T as GatTrait>::Assoc<'a>: Send,
{}

async fn test_gat() {
    let x = GatBar(());
    async {}.await;
    drop(x);
}

fn main() {
    assert_send(test_dyn_fn());
    assert_send(test_hrtb());
    assert_send(test_gat());
}
