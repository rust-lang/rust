// Check that we properly record borrows when we are doing an
// overloaded, autoderef of a value obtained via an overloaded index
// operator. The accounting of the all the implicit things going on
// here is rather subtle. Issue #20232.

use std::ops::{Deref, Index};

struct MyVec<T> { x: T }

impl<T> Index<usize> for MyVec<T> {
    type Output = T;
    fn index(&self, _: usize) -> &T {
        &self.x
    }
}

struct MyPtr<T> { x: T }

impl<T> Deref for MyPtr<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.x
    }
}

struct Foo { f: usize }

fn main() {
    let mut v = MyVec { x: MyPtr { x: Foo { f: 22 } } };
    let i = &v[0].f;
    v = MyVec { x: MyPtr { x: Foo { f: 23 } } };
    //~^ ERROR cannot assign to `v` because it is borrowed
    read(*i);
}

fn read(_: usize) { }
