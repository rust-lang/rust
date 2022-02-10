#![feature(no_core, intrinsics, lang_items)]
#![crate_type="rlib"]
#![no_core]

extern "rust-intrinsic" {
    fn atomic_xadd<T>(dst: *mut T, src: T) -> T;
}

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}
#[lang = "freeze"]
trait Freeze {}

impl<T: ?Sized> Copy for *mut T {}

#[cfg(target_has_atomic = "8")]
pub unsafe fn atomic_u8(x: *mut u8) {
    atomic_xadd(x, 1);
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "8")]
pub unsafe fn atomic_i8(x: *mut i8) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "16")]
pub unsafe fn atomic_u16(x: *mut u16) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "16")]
pub unsafe fn atomic_i16(x: *mut i16) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "32")]
pub unsafe fn atomic_u32(x: *mut u32) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "32")]
pub unsafe fn atomic_i32(x: *mut i32) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "64")]
pub unsafe fn atomic_u64(x: *mut u64) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "64")]
pub unsafe fn atomic_i64(x: *mut i64) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "128")]
pub unsafe fn atomic_u128(x: *mut u128) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "128")]
pub unsafe fn atomic_i128(x: *mut i128) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "ptr")]
pub unsafe fn atomic_usize(x: *mut usize) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "ptr")]
pub unsafe fn atomic_isize(x: *mut isize) {
    atomic_xadd(x, 1);
}
