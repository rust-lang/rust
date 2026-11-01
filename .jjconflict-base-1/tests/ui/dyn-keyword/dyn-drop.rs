#![deny(dyn_drop)]
fn foo(_: Box<dyn Drop>) {} //~ ERROR: types that do not implement `Drop` can still have drop glue,
fn bar(_: &dyn Drop) {} //~ ERROR: types that do not implement `Drop` can still have drop glue,
fn baz(_: *mut dyn Drop) {} //~ ERROR: types that do not implement `Drop` can still have drop glue,
struct Foo {
    _x: Box<dyn Drop>, //~ ERROR: types that do not implement `Drop` can still have drop glue,
}
trait Bar {
    type T: ?Sized;
}
struct Baz {}
impl Bar for Baz {
    type T = dyn Drop; //~ ERROR: types that do not implement `Drop` can still have drop glue,
}
fn main() {}
