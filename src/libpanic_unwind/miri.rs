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
