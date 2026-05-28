//@ check-pass

// Regression test for #54378.

#![feature(never_type)]

use std::marker::PhantomData;

pub trait Machine<'a, 'mir, 'tcx>: Sized {
    type MemoryKinds: ::std::fmt::Debug + Copy + Eq;
    const MUT_STATIC_KIND: Option<Self::MemoryKinds>;
}

pub struct CompileTimeEvaluator<'a, 'mir, 'tcx: 'a+'mir> {
    pub _data: PhantomData<(&'a (), &'mir (), &'tcx ())>,
}

impl<'a, 'mir, 'tcx: 'a + 'mir> Machine<'a, 'mir, 'tcx>
    for CompileTimeEvaluator<'a, 'mir, 'tcx>
{
    type MemoryKinds = !;

    const MUT_STATIC_KIND: Option<!> = None;
}

fn main() {}
