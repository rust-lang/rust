// build-pass (FIXME(62277): could be check-pass?)

#![feature(fn_traits, unboxed_closures)]

#[allow(dead_code)]
struct Foo;

impl Fn<(&(),)> for Foo {
    extern "rust-call" fn call(&self, (_,): (&(),)) {}
}

impl FnMut<(&(),)> for Foo {
    extern "rust-call" fn call_mut(&mut self, (_,): (&(),)) {}
}

impl FnOnce<(&(),)> for Foo {
    type Output = ();

    extern "rust-call" fn call_once(self, (_,): (&(),)) {}
}

fn main() {}
