// Regression test for the ICE caused by the example in
// https://github.com/rust-lang/rust/issues/86053#issuecomment-855672258

#![feature(c_variadic)]

trait H<T> {}

unsafe extern "C" fn ordering4<'a, F: H<&'static &'a ()>>(_: (), ...) {}
//~^ ERROR: in type `&'static &'a ()`, reference has a longer lifetime than the data it references [E0491]

fn main() {}
