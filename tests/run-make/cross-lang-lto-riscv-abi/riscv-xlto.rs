#![allow(internal_features)]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "pointee_sized"]
trait PointeeSized {}
#[lang = "size_of_val"]
trait MetaSized: PointeeSized {}
#[lang = "sized"]
trait Sized: MetaSized {}

#[no_mangle]
pub fn hello() {}
