// Regression test for #85945: Don't suggest `?Sized` bound if an explicit
// `Sized` bound is already in a `where` clause.
fn foo<T>(_: &T) where T: Sized {}
fn bar() { foo(""); }
//~^ERROR the size for values of type

pub fn main() {
}
