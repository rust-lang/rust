//@ run-pass
#![allow(unused_variables)]

// Test slicing sugar.

extern crate core;
use core::ops::{Index, IndexMut, Range, RangeTo, RangeFrom, RangeFull};
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNT: AtomicUsize = AtomicUsize::new(0);

struct Foo;

impl Index<Range<Foo>> for Foo {
    type Output = Foo;
    fn index(&self, index: Range<Foo>) -> &Foo {
        COUNT.fetch_add(1, Ordering::Relaxed);
        self
    }
}
impl Index<RangeTo<Foo>> for Foo {
    type Output = Foo;
    fn index(&self, index: RangeTo<Foo>) -> &Foo {
        COUNT.fetch_add(1, Ordering::Relaxed);
        self
    }
}
impl Index<RangeFrom<Foo>> for Foo {
    type Output = Foo;
    fn index(&self, index: RangeFrom<Foo>) -> &Foo {
        COUNT.fetch_add(1, Ordering::Relaxed);
        self
    }
}
impl Index<RangeFull> for Foo {
    type Output = Foo;
    fn index(&self, _index: RangeFull) -> &Foo {
        COUNT.fetch_add(1, Ordering::Relaxed);
        self
    }
}

impl IndexMut<Range<Foo>> for Foo {
    fn index_mut(&mut self, index: Range<Foo>) -> &mut Foo {
        COUNT.fetch_add(1, Ordering::Relaxed);
        self
    }
}
impl IndexMut<RangeTo<Foo>> for Foo {
    fn index_mut(&mut self, index: RangeTo<Foo>) -> &mut Foo {
        COUNT.fetch_add(1, Ordering::Relaxed);
        self
    }
}
impl IndexMut<RangeFrom<Foo>> for Foo {
    fn index_mut(&mut self, index: RangeFrom<Foo>) -> &mut Foo {
        COUNT.fetch_add(1, Ordering::Relaxed);
        self
    }
}
impl IndexMut<RangeFull> for Foo {
    fn index_mut(&mut self, _index: RangeFull) -> &mut Foo {
        COUNT.fetch_add(1, Ordering::Relaxed);
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
    assert_eq!(COUNT.load(Ordering::Relaxed), 8);
}
