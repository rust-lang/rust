#![allow(internal_features)]
#![feature(no_core, lang_items)]
#![no_core]
#![crate_type = "rlib"]

#[lang = "pointee_sized"]
trait PointeeSized {}
#[lang = "size_of_val"]
trait MetaSized: PointeeSized {}
#[lang = "sized"]
trait Sized: MetaSized {}

pub fn foo() {}
