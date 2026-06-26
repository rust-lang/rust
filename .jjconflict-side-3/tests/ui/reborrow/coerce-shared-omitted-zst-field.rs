#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

struct NonCopyZst;

#[derive(Clone, Copy)]
struct CopyZst;

struct ReborrowZst<'a>(PhantomData<&'a mut ()>);

impl Reborrow for ReborrowZst<'_> {}

struct Source<'a> {
    value: &'a mut i32,
    marker: NonCopyZst,
    //~^ ERROR the trait bound `NonCopyZst: Copy` is not satisfied
    //~| ERROR implementing `CoerceShared` requires source fields omitted from the target
}

impl Reborrow for Source<'_> {}

#[derive(Clone, Copy)]
struct Target<'a> {
    value: &'a i32,
}

impl<'a> CoerceShared<Target<'a>> for Source<'a> {}

struct CopyZstSource<'a> {
    value: &'a mut i32,
    marker: CopyZst,
}

impl Reborrow for CopyZstSource<'_> {}

#[derive(Clone, Copy)]
struct CopyZstTarget<'a> {
    value: &'a i32,
}

impl<'a> CoerceShared<CopyZstTarget<'a>> for CopyZstSource<'a> {}

struct ReborrowZstSource<'a> {
    value: &'a mut i32,
    marker: ReborrowZst<'a>,
}

impl Reborrow for ReborrowZstSource<'_> {}

#[derive(Clone, Copy)]
struct ReborrowZstTarget<'a> {
    value: &'a i32,
}

impl<'a> CoerceShared<ReborrowZstTarget<'a>> for ReborrowZstSource<'a> {}

fn main() {}
