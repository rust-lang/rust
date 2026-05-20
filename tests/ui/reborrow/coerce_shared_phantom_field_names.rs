//@ run-pass

// Regression test: CoerceShared borrowck must filter PhantomData fields
// before matching by name, consistent with coherence's
// `collect_struct_data_fields`. Without the filter, differently-named
// PhantomData fields would cause lifetime constraints to be silently dropped.

#![feature(reborrow)]
use std::marker::{CoerceShared, PhantomData, Reborrow};

struct Source<'a> {
    data: &'a mut i32,
    _marker: PhantomData<&'a ()>,
}
impl<'a> Reborrow for Source<'a> {}

#[derive(Clone, Copy)]
struct Dest<'a> {
    data: &'a i32,
    _lifetime: PhantomData<&'a ()>, // Different name from Source's PhantomData field
}
impl<'a> CoerceShared<Dest<'a>> for Source<'a> {}

fn use_ref<'a>(s: Dest<'a>) -> &'a i32 {
    s.data
}

fn main() {
    let mut val = 42;
    let s = Source { data: &mut val, _marker: PhantomData };
    let r = use_ref(s);
    assert_eq!(*r, 42);
}
