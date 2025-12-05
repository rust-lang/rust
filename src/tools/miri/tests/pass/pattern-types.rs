#![feature(pattern_types, pattern_type_macro, sized_hierarchy)]
#![allow(dead_code)]

use std::marker::PointeeSized;
use std::mem::transmute;

pub struct NonNull<T: PointeeSized> {
    pointer: std::pat::pattern_type!(*const T is !null),
}

trait Trait {}
impl Trait for () {}

fn main() {
    unsafe {
        let _: NonNull<dyn Trait> = NonNull { pointer: transmute(&mut () as *mut dyn Trait) };
    }
}
