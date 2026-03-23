//@ compile-flags: --crate-type=lib
#![feature(sized_hierarchy)]

use std::marker::{SizeOfVal, PointeeSized};

pub trait Tr: PointeeSized {}
impl Tr for u32 {}

pub fn sized() -> Box<impl Tr + Sized> {
    if true {
        let x = sized();
        let y: Box<dyn Tr> = x;
    }
    Box::new(1u32)
}

pub fn neg_sized() -> Box<impl Tr + ?Sized> {
    if true {
        let x = neg_sized();
        let y: Box<dyn Tr> = x;
//~^ ERROR: the size for values of type `impl Tr + SizeOfVal` cannot be known
    }
    Box::new(1u32)
}

pub fn sizeofval() -> Box<impl Tr + SizeOfVal> {
    if true {
        let x = sizeofval();
        let y: Box<dyn Tr> = x;
//~^ ERROR: the size for values of type `impl Tr + SizeOfVal` cannot be known
    }
    Box::new(1u32)
}

pub fn pointeesized() -> Box<impl Tr + PointeeSized> {
//~^ ERROR: the size for values of type `impl Tr + PointeeSized` cannot be known
    if true {
        let x = pointeesized();
//~^ ERROR: the size for values of type `impl Tr + PointeeSized` cannot be known
        let y: Box<dyn Tr> = x;
//~^ ERROR: the size for values of type `impl Tr + PointeeSized` cannot be known
//~| ERROR: the size for values of type `impl Tr + PointeeSized` cannot be known
    }
    Box::new(1u32)
}
