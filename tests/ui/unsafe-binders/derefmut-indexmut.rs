//@ run-pass

#![feature(unsafe_binders)]

use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::unsafe_binder::{unwrap_binder, wrap_binder};

struct Foo(unsafe<'a> &'a u32);

impl Deref for Foo {
    type Target = unsafe<'a> &'a u32;

    fn deref(&self) -> &unsafe<'a> &'a u32 {
        &self.0
    }
}

impl DerefMut for Foo {
    fn deref_mut(&mut self) -> &mut unsafe<'a> &'a u32 {
        &mut self.0
    }
}

impl Index<()> for Foo {
    type Output = unsafe<'a> &'a u32;

    fn index(&self, _: ()) -> &unsafe<'a> &'a u32 {
        &self.0
    }
}

impl IndexMut<()> for Foo {
    fn index_mut(&mut self, _: ()) -> &mut unsafe<'a> &'a u32 {
        &mut self.0
    }
}

fn main() {
    let mut foo = Foo(unsafe { wrap_binder!(&17) });
    let _m = unsafe { &mut unwrap_binder!(*foo) };
    let _m = unsafe { &mut unwrap_binder!(foo[()]) };
}
