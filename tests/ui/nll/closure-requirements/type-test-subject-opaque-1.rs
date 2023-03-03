// check-fail
// known-bug: #107426

use std::marker::PhantomData;
#[derive(Clone, Copy)]
pub struct Scope<'a>(&'a PhantomData<&'a mut &'a ()>);
fn event<'a, F: FnMut() + 'a>(_: Scope<'a>, _: F) {}
fn make_fn<'a>(_: Scope<'a>) -> impl Fn() + Copy + 'a {
    || {}
}

fn foo(cx: Scope) {
    let open_toggle = make_fn(cx);

    || event(cx, open_toggle);
}

fn main() {}
