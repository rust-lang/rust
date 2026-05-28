//! Regression test for https://github.com/rust-lang/rust/issues/14399

//@ run-pass
// #14399
// We'd previously ICE if we had a method call whose return
// value was coerced to a trait object. (v.clone() returns Box<B1>
// which is coerced to Box<A>).


#[derive(Clone)]
struct B1;

trait A { fn foo(&self) {} } //~ WARN method `foo` is never used
impl A for B1 {}

fn main() {
    let v: Box<_> = Box::new(B1);
    let _c: Box<dyn A> = v.clone();
}
