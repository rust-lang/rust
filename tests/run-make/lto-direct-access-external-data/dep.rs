#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

#[lang = "pointee_sized"]
trait PointeeSized {}
#[lang = "meta_sized"]
trait MetaSized: PointeeSized {}
#[lang = "sized"]
trait Sized: MetaSized {}

#[lang = "copy"]
pub trait Copy {}

impl Copy for i32 {}

unsafe extern "C" {
    pub safe static VAR: i32;
}

#[no_mangle]
pub fn refer_dep() -> i32 {
    unsafe { VAR }
}
