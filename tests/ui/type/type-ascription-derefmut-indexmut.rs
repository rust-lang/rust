//@ run-pass

#![feature(type_ascription)]

use std::ops::{Deref, DerefMut, Index, IndexMut};

struct Foo(u32);

impl Deref for Foo {
    type Target = u32;

    fn deref(&self) -> &u32 {
        &self.0
    }
}

impl DerefMut for Foo {
    fn deref_mut(&mut self) -> &mut u32 {
        &mut self.0
    }
}

impl Index<()> for Foo {
    type Output = u32;

    fn index(&self, _: ()) -> &u32 {
        &self.0
    }
}

impl IndexMut<()> for Foo {
    fn index_mut(&mut self, _: ()) -> &mut u32 {
        &mut self.0
    }
}

fn main() {
    let _m = &mut type_ascribe!(*Foo(17), u32);
    let _m = &mut type_ascribe!(Foo(17)[()], u32);
}
