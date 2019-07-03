// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

fn foo<T: 'static>(_: T) {}

fn bar<T>(x: &'static T) {
    foo(x);
}
fn main() {}
