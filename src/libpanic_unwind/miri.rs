use core::any::Any;
use alloc::boxed::Box;

pub fn payload() -> *mut u8 {
    core::ptr::null_mut()
}

pub unsafe fn panic(data: Box<dyn Any + Send>) -> ! {
    let raw_val = core::mem::transmute::<_, u128>(Box::into_raw(data));
    core::intrinsics::miri_start_panic(raw_val)
}

pub unsafe fn cleanup(ptr: *mut u8) -> Box<dyn Any + Send> {
    Box::from_raw(ptr)
}

