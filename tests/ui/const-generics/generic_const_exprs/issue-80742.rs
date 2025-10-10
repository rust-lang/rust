//@ check-fail
//@ known-bug: #97477
//@ failure-status: 101
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "thread 'rustc'.*panicked.*\n" -> ""
//@ normalize-stderr: "(error: internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "
//@ rustc-env:RUST_BACKTRACE=0

// This test used to cause an ICE in rustc_mir::interpret::step::eval_rvalue_into_place

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::size_of;

struct Inline<T>
where
    [u8; size_of::<T>() + 1]: ,
{
    _phantom: PhantomData<T>,
    buf: [u8; size_of::<T>() + 1],
}

impl<T> Inline<T>
where
    [u8; size_of::<T>() + 1]: ,
{
    pub fn new(val: T) -> Inline<T> {
        todo!()
    }
}

fn main() {
    let dst = Inline::<dyn Debug>::new(0);
}
