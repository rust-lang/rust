//@ check-pass

//! Accept lifetime extensions of un-exercised lifetimes.

#![feature(transmutability, core_intrinsics)]

use std::mem::{Assume, TransmuteFrom};

unsafe fn transmute<Src, Dst>(src: Src) -> Dst
where
    Dst: TransmuteFrom<Src, { Assume::SAFETY }>,
{
    core::intrinsics::transmute_unchecked(src)
}

enum Void {}

mod phantom {
    use super::*;
    use std::marker::PhantomData;

    fn extend_bare<'a>(src: PhantomData<&'a u8>) -> PhantomData<&'static u8> {
        unsafe { transmute(src) }
    }
}


mod tuple {
    use super::*;

    fn extend_pair<'a>(src: (&'a u8, Void)) -> (&'static u8, Void) {
        unsafe { transmute(src) }
    }
}

mod r#struct {
    use super::*;

    struct Struct<'a>(&'a u8, Void);

    fn extend_struct<'a>(src: Struct<'a>) -> Struct<'static> {
        unsafe { transmute(src) }
    }
}

mod r#enum {
    use super::*;

    enum Single<'a> {
        A(&'a u8, Void),
    }

    fn extend_single<'a>(src: Single<'a>) -> Single<'static> {
        unsafe { transmute(src) }
    }

    enum Multi<'a> {
        A(&'a u8, Void),
        B,
        C,
    }

    fn extend_multi<'a>(src: Multi<'a>) -> Multi<'static> {
        unsafe { transmute(src) }
    }
}

fn main() {}
