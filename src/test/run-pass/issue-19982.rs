// pretty-expanded FIXME #23616

#![feature(fn_traits, unboxed_closures)]

#[allow(dead_code)]
struct Foo;

impl<'a> Fn<(&'a (),)> for Foo {
    extern "rust-call" fn call(&self, (_,): (&(),)) {}
}

impl<'a> FnMut<(&'a (),)> for Foo {
    extern "rust-call" fn call_mut(&mut self, (_,): (&(),)) {}
}

impl<'a> FnOnce<(&'a (),)> for Foo {
    type Output = ();

    extern "rust-call" fn call_once(self, (_,): (&(),)) {}
}

fn main() {}
