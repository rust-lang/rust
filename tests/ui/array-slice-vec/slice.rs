//@ run-pass
#![allow(unused_variables)]

// Test slicing sugar.

// FIXME(static_mut_refs): this could use an atomic
#![allow(static_mut_refs)]

extern crate core;
use core::ops::{Index, IndexMut, Range, RangeTo, RangeFrom, RangeFull};

static mut COUNT: usize = 0;

struct Foo;

impl Index<Range<Foo>> for Foo {
    type Output = Foo;
    fn index(&self, index: Range<Foo>) -> &Foo {
        unsafe { COUNT += 1; }
        self
    }
}
impl Index<RangeTo<Foo>> for Foo {
    type Output = Foo;
    fn index(&self, index: RangeTo<Foo>) -> &Foo {
        unsafe { COUNT += 1; }
        self
    }
}
impl Index<RangeFrom<Foo>> for Foo {
    type Output = Foo;
    fn index(&self, index: RangeFrom<Foo>) -> &Foo {
        unsafe { COUNT += 1; }
        self
    }
}
impl Index<RangeFull> for Foo {
    type Output = Foo;
    fn index(&self, _index: RangeFull) -> &Foo {
        unsafe { COUNT += 1; }
        self
    }
}

impl IndexMut<Range<Foo>> for Foo {
    fn index_mut(&mut self, index: Range<Foo>) -> &mut Foo {
        unsafe { COUNT += 1; }
        self
    }
}
impl IndexMut<RangeTo<Foo>> for Foo {
    fn index_mut(&mut self, index: RangeTo<Foo>) -> &mut Foo {
        unsafe { COUNT += 1; }
        self
    }
}
impl IndexMut<RangeFrom<Foo>> for Foo {
    fn index_mut(&mut self, index: RangeFrom<Foo>) -> &mut Foo {
        unsafe { COUNT += 1; }
        self
    }
}
impl IndexMut<RangeFull> for Foo {
    fn index_mut(&mut self, _index: RangeFull) -> &mut Foo {
        unsafe { COUNT += 1; }
        self
    }
}


fn main() {
    let mut x = Foo;
    let _ = &x[..];
    let _ = &x[Foo..];
    let _ = &x[..Foo];
    let _ = &x[Foo..Foo];
    let _ = &mut x[..];
    let _ = &mut x[Foo..];
    let _ = &mut x[..Foo];
    let _ = &mut x[Foo..Foo];
    unsafe {
        assert_eq!(COUNT, 8);
    }
}
