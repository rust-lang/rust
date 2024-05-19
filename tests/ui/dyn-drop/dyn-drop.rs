#![deny(dyn_drop)]
#![allow(bare_trait_objects)]
fn foo(_: Box<dyn Drop>) {} //~ ERROR
fn bar(_: &dyn Drop) {} //~ERROR
fn baz(_: *mut Drop) {} //~ ERROR
struct Foo {
  _x: Box<dyn Drop> //~ ERROR
}
trait Bar {
  type T: ?Sized;
}
struct Baz {}
impl Bar for Baz {
  type T = dyn Drop; //~ ERROR
}
fn main() {}
