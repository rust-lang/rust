// run-pass
// pretty-expanded FIXME #23616

#![feature(fn_traits, unboxed_closures)]

trait Foo { fn dummy(&self) { }}

struct Bar;

impl<'a> std::ops::Fn<(&'a (dyn Foo+'a),)> for Bar {
    extern "rust-call" fn call(&self, _: (&'a dyn Foo,)) {}
}

impl<'a> std::ops::FnMut<(&'a (dyn Foo+'a),)> for Bar {
    extern "rust-call" fn call_mut(&mut self, a: (&'a dyn Foo,)) { self.call(a) }
}

impl<'a> std::ops::FnOnce<(&'a (dyn Foo+'a),)> for Bar {
    type Output = ();
    extern "rust-call" fn call_once(self, a: (&'a dyn Foo,)) { self.call(a) }
}

struct Baz;

impl Foo for Baz {}

fn main() {
    let bar = Bar;
    let baz = &Baz;
    bar(baz);
}
