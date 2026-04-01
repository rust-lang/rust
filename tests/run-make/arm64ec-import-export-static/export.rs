#![crate_type = "dylib"]
#![allow(internal_features)]
#![feature(no_core, lang_items)]
#![no_core]
#![no_std]

// This is needed because of #![no_core]:
#[lang = "pointee_sized"]
pub trait PointeeSized {}
#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}
#[lang = "sized"]
pub trait Sized: MetaSized {}
#[lang = "sync"]
trait Sync {}
impl Sync for i32 {}
#[lang = "copy"]
pub trait Copy {}
impl Copy for i32 {}
#[lang = "drop_in_place"]
pub unsafe fn drop_in_place<T: ?Sized>(_: *mut T) {}
#[no_mangle]
extern "system" fn _DllMainCRTStartup(_: *const u8, _: u32, _: *const u8) -> u32 {
    1
}

pub static VALUE: i32 = 42;
