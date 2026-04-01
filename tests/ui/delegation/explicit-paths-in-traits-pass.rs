//@ run-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait ToReuse {
    fn foo(&self, x: i32) -> i32 { x }
    fn foo1(x: i32) -> i32 { x }
}

fn foo2() -> i32 { 42 }

trait Trait: ToReuse {
    reuse ToReuse::foo;
    reuse <Self as ToReuse>::foo1;
    reuse foo2;
}

struct S;
impl ToReuse for S {}
impl Trait for S {}

fn main() {
    assert_eq!(<S as Trait>::foo(&S, 1), 1);
    assert_eq!(<S as Trait>::foo1(1), 1);
    assert_eq!(<S as Trait>::foo2(), 42);
}
