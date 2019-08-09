// run-pass
// #14399
// We'd previously ICE if we had a method call whose return
// value was coerced to a trait object. (v.clone() returns Box<B1>
// which is coerced to Box<A>).

// pretty-expanded FIXME #23616

#![feature(box_syntax)]

#[derive(Clone)]
struct B1;

trait A { fn foo(&self) {} }
impl A for B1 {}

fn main() {
    let v: Box<_> = box B1;
    let _c: Box<dyn A> = v.clone();
}
