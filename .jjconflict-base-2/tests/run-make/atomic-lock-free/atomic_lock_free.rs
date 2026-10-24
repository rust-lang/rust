#![feature(no_core, intrinsics, lang_items)]
#![feature(adt_const_params)]
#![crate_type = "rlib"]
#![no_core]

pub enum AtomicOrdering {
    // These values must match the compiler's `AtomicOrdering` defined in
    // `rustc_middle/src/ty/consts/int.rs`!
    Relaxed = 0,
    Release = 1,
    Acquire = 2,
    AcqRel = 3,
    SeqCst = 4,
}

#[rustc_intrinsic]
unsafe fn atomic_xadd<T, U, const ORD: AtomicOrdering>(dst: *mut T, src: U) -> T;

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}
#[lang = "copy"]
trait Copy {}
#[lang = "freeze"]
trait Freeze {}
#[lang = "const_param_ty"]
pub trait ConstParamTy {}

impl<T: ?Sized> Copy for *mut T {}
impl ConstParamTy for AtomicOrdering {}

#[cfg(target_has_atomic = "8")]
#[unsafe(no_mangle)] // let's make sure we actually generate a symbol to check
pub unsafe fn atomic_u8(x: *mut u8) {
    atomic_xadd::<_, _, { AtomicOrdering::SeqCst }>(x, 1u8);
}
#[cfg(target_has_atomic = "8")]
#[unsafe(no_mangle)]
pub unsafe fn atomic_i8(x: *mut i8) {
    atomic_xadd::<_, _, { AtomicOrdering::SeqCst }>(x, 1i8);
}
#[cfg(target_has_atomic = "16")]
#[unsafe(no_mangle)]
pub unsafe fn atomic_u16(x: *mut u16) {
    atomic_xadd::<_, _, { AtomicOrdering::SeqCst }>(x, 1u16);
}
#[cfg(target_has_atomic = "16")]
#[unsafe(no_mangle)]
pub unsafe fn atomic_i16(x: *mut i16) {
    atomic_xadd::<_, _, { AtomicOrdering::SeqCst }>(x, 1i16);
}
#[cfg(target_has_atomic = "32")]
#[unsafe(no_mangle)]
pub unsafe fn atomic_u32(x: *mut u32) {
    atomic_xadd::<_, _, { AtomicOrdering::SeqCst }>(x, 1u32);
}
#[cfg(target_has_atomic = "32")]
#[unsafe(no_mangle)]
pub unsafe fn atomic_i32(x: *mut i32) {
    atomic_xadd::<_, _, { AtomicOrdering::SeqCst }>(x, 1i32);
}
#[cfg(target_has_atomic = "64")]
#[unsafe(no_mangle)]
pub unsafe fn atomic_u64(x: *mut u64) {
    atomic_xadd::<_, _, { AtomicOrdering::SeqCst }>(x, 1u64);
}
#[cfg(target_has_atomic = "64")]
#[unsafe(no_mangle)]
pub unsafe fn atomic_i64(x: *mut i64) {
    atomic_xadd::<_, _, { AtomicOrdering::SeqCst }>(x, 1i64);
}
#[cfg(target_has_atomic = "128")]
#[unsafe(no_mangle)]
pub unsafe fn atomic_u128(x: *mut u128) {
    atomic_xadd::<_, _, { AtomicOrdering::SeqCst }>(x, 1u128);
}
#[cfg(target_has_atomic = "128")]
#[unsafe(no_mangle)]
pub unsafe fn atomic_i128(x: *mut i128) {
    atomic_xadd::<_, _, { AtomicOrdering::SeqCst }>(x, 1i128);
}
#[cfg(target_has_atomic = "ptr")]
#[unsafe(no_mangle)]
pub unsafe fn atomic_usize(x: *mut usize) {
    atomic_xadd::<_, _, { AtomicOrdering::SeqCst }>(x, 1usize);
}
#[cfg(target_has_atomic = "ptr")]
#[unsafe(no_mangle)]
pub unsafe fn atomic_isize(x: *mut isize) {
    atomic_xadd::<_, _, { AtomicOrdering::SeqCst }>(x, 1isize);
}
