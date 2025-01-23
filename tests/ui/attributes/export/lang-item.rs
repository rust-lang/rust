#![feature(no_core, lang_items, export_stable)]
#![allow(incomplete_features)]
#![crate_type = "sdylib"]
#![no_core]

#[lang = "pointee_sized"]
//~^ ERROR lang items are not allowed in stable dylibs
pub trait PointeeSized {}

#[lang = "meta_sized"]
//~^ ERROR lang items are not allowed in stable dylibs
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
//~^ ERROR lang items are not allowed in stable dylibs
trait Sized {}
