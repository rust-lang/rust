//@ compile-flags: --crate-type=lib
#![feature(sized_hierarchy)]

use std::marker::{MetaSized, PointeeSized};

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
//~^ ERROR: the size for values of type `impl Tr + MetaSized` cannot be known
    }
    Box::new(1u32)
}

pub fn metasized() -> Box<impl Tr + MetaSized> {
    if true {
        let x = metasized();
        let y: Box<dyn Tr> = x;
//~^ ERROR: the size for values of type `impl Tr + MetaSized` cannot be known
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
