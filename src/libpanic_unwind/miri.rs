#![allow(nonstandard_style)]

use core::any::Any;
use alloc::boxed::Box;

pub fn payload() -> *mut u8 {
    core::ptr::null_mut()
}

pub unsafe fn panic(data: Box<dyn Any + Send>) -> ! {
    core::intrinsics::miri_start_panic(Box::into_raw(data))
}

pub unsafe fn cleanup(ptr: *mut u8) -> Box<dyn Any + Send> {
    Box::from_raw(ptr)
}

// This is required by the compiler to exist (e.g., it's a lang item),
// but is never used by Miri. Therefore, we just use a stub here
#[lang = "eh_personality"]
#[cfg(not(test))]
fn rust_eh_personality() {
    unsafe { core::intrinsics::abort() }
}

// The rest is required on *some* targets to exist (specifically, MSVC targets that use SEH).
// We just add it on all targets. Copied from `seh.rs`.
#[repr(C)]
pub struct _TypeDescriptor {
    pub pVFTable: *const u8,
    pub spare: *mut u8,
    pub name: [u8; 11],
}

const TYPE_NAME: [u8; 11] = *b"rust_panic\0";

#[cfg_attr(not(test), lang = "eh_catch_typeinfo")]
static mut TYPE_DESCRIPTOR: _TypeDescriptor = _TypeDescriptor {
    pVFTable: core::ptr::null(),
    spare: core::ptr::null_mut(),
    name: TYPE_NAME,
};
