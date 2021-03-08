// run-rustfix

#![warn(clippy::use_self)]
#![allow(dead_code)]
#![allow(clippy::should_implement_trait, clippy::boxed_local)]

use std::ops::Mul;

trait SelfTrait {
    fn refs(p1: &Self) -> &Self;
    fn ref_refs<'a>(p1: &'a &'a Self) -> &'a &'a Self;
    fn mut_refs(p1: &mut Self) -> &mut Self;
    fn nested(p1: Box<Self>, p2: (&u8, &Self));
    fn vals(r: Self) -> Self;
}

#[derive(Default)]
struct Bad;

impl SelfTrait for Bad {
    fn refs(p1: &Bad) -> &Bad {
        p1
    }

    fn ref_refs<'a>(p1: &'a &'a Bad) -> &'a &'a Bad {
        p1
    }

    fn mut_refs(p1: &mut Bad) -> &mut Bad {
        p1
    }

    fn nested(_p1: Box<Bad>, _p2: (&u8, &Bad)) {}

    fn vals(_: Bad) -> Bad {
        Bad::default()
    }
}

impl Mul for Bad {
    type Output = Bad;

    fn mul(self, rhs: Bad) -> Bad {
        rhs
    }
}

impl Clone for Bad {
    fn clone(&self) -> Self {
        // FIXME: applicable here
        Bad
    }
}

#[derive(Default)]
struct Good;

impl SelfTrait for Good {
    fn refs(p1: &Self) -> &Self {
        p1
    }

    fn ref_refs<'a>(p1: &'a &'a Self) -> &'a &'a Self {
        p1
    }

    fn mut_refs(p1: &mut Self) -> &mut Self {
        p1
    }

    fn nested(_p1: Box<Self>, _p2: (&u8, &Self)) {}

    fn vals(_: Self) -> Self {
        Self::default()
    }
}

impl Mul for Good {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        rhs
    }
}

trait NameTrait {
    fn refs(p1: &u8) -> &u8;
    fn ref_refs<'a>(p1: &'a &'a u8) -> &'a &'a u8;
    fn mut_refs(p1: &mut u8) -> &mut u8;
    fn nested(p1: Box<u8>, p2: (&u8, &u8));
    fn vals(p1: u8) -> u8;
}

// Using `Self` instead of the type name is OK
impl NameTrait for u8 {
    fn refs(p1: &Self) -> &Self {
        p1
    }

    fn ref_refs<'a>(p1: &'a &'a Self) -> &'a &'a Self {
        p1
    }

    fn mut_refs(p1: &mut Self) -> &mut Self {
        p1
    }

    fn nested(_p1: Box<Self>, _p2: (&Self, &Self)) {}

    fn vals(_: Self) -> Self {
        Self::default()
    }
}

fn main() {}
