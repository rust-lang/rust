#![allow(internal_features)]
#![feature(lang_items, no_core)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}
#[lang = "size_of_val"]
pub trait SizeOfVal: PointeeSized {}
#[lang = "sized"]
pub trait Sized: SizeOfVal {}

pub fn f() {}
