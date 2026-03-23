#![allow(internal_features)]
#![feature(no_core, lang_items)]
#![no_core]
#![crate_type = "rlib"]

#[lang = "pointee_sized"]
trait PointeeSized {}
#[lang = "size_of_val"]
trait SizeOfVal: PointeeSized {}
#[lang = "sized"]
trait Sized: SizeOfVal {}

pub fn foo() {}
