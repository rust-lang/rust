// run-pass
// Test using overloaded indexing when the "map" is stored in a
// field. This caused problems at some point.

use std::ops::Index;

struct Foo {
    x: isize,
    y: isize,
}

struct Bar {
    foo: Foo
}

impl Index<isize> for Foo {
    type Output = isize;

    fn index(&self, z: isize) -> &isize {
        if z == 0 {
            &self.x
        } else {
            &self.y
        }
    }
}

trait Int {
    fn get(self) -> isize;
    fn get_from_ref(&self) -> isize;
    fn inc(&mut self);
}

impl Int for isize {
    fn get(self) -> isize { self }
    fn get_from_ref(&self) -> isize { *self }
    fn inc(&mut self) { *self += 1; }
}

fn main() {
    let f = Bar { foo: Foo {
        x: 1,
        y: 2,
    } };
    assert_eq!(f.foo[1].get(), 2);
}
