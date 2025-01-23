#![feature(no_core, lang_items, freeze_impls)]
#![crate_type = "rlib"]
#![no_core]

pub static STATIC_BOOL: bool = true;

pub static mut STATIC_MUT_BOOL: bool = true;

const CONST_BOOL: bool = true;
pub static CONST_BOOL_REF: &'static bool = &CONST_BOOL;

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[lang = "copy"]
trait Copy {}
impl Copy for bool {}
impl Copy for &bool {}

#[lang = "freeze"]
trait Freeze {}

// No `UnsafeCell`, so everything is `Freeze`.
impl<T: ?Sized> Freeze for T {}

#[lang = "sync"]
trait Sync {}
impl Sync for bool {}
impl Sync for &'static bool {}

#[lang = "drop_in_place"]
pub unsafe fn drop_in_place<T: ?Sized>(_: *mut T) {}
