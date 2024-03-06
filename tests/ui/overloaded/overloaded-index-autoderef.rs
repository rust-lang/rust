//@ run-pass
#![allow(stable_features)]

// Test overloaded indexing combined with autoderef.

use std::ops::{Index, IndexMut};

struct Foo {
    x: isize,
    y: isize,
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

impl IndexMut<isize> for Foo {
    fn index_mut(&mut self, z: isize) -> &mut isize {
        if z == 0 {
            &mut self.x
        } else {
            &mut self.y
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
    let mut f: Box<_> = Box::new(Foo {
        x: 1,
        y: 2,
    });

    assert_eq!(f[1], 2);

    f[0] = 3;

    assert_eq!(f[0], 3);

    // Test explicit IndexMut where `f` must be autoderef:
    {
        let p = &mut f[1];
        *p = 4;
    }

    // Test explicit Index where `f` must be autoderef:
    {
        let p = &f[1];
        assert_eq!(*p, 4);
    }

    // Test calling methods with `&mut self`, `self, and `&self` receivers:
    f[1].inc();
    assert_eq!(f[1].get(), 5);
    assert_eq!(f[1].get_from_ref(), 5);
}
