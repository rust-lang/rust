// run-pass
// Test that when we write `x.foo()`, we do not have to know the
// complete type of `x` in order to type-check the method call. In
// this case, we know that `x: Vec<_1>`, but we don't know what type
// `_1` is (because the call to `push` comes later). To pick between
// the impls, we would have to know `_1`, since we have to know
// whether `_1: MyCopy` or `_1 == Box<i32>`.  However (and this is the
// point of the test), we don't have to pick between the two impls --
// it is enough to know that `foo` comes from the `Foo` trait. We can
// codegen the call as `Foo::foo(&x)` and let the specific impl get
// chosen later.

#![feature(box_syntax)]

trait Foo {
    fn foo(&self) -> isize;
}

trait MyCopy { fn foo(&self) { } }
impl MyCopy for i32 { }

impl<T:MyCopy> Foo for Vec<T> {
    fn foo(&self) -> isize {1}
}

impl Foo for Vec<Box<i32>> {
    fn foo(&self) -> isize {2}
}

fn call_foo_copy() -> isize {
    let mut x = Vec::new();
    let y = x.foo();
    x.push(0_i32);
    y
}

fn call_foo_other() -> isize {
    let mut x: Vec<_> = Vec::new();
    let y = x.foo();
    let z: Box<i32> = box 0;
    x.push(z);
    y
}

fn main() {
    assert_eq!(call_foo_copy(), 1);
    assert_eq!(call_foo_other(), 2);
}
