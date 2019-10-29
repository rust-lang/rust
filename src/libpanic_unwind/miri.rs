pub fn payload() -> *mut u8 {
    core::ptr::null_mut()
}

pub unsafe fn panic(data: Box<dyn Any + Send>) -> u32 {
    core::intrinsics::miri_start_panic(Box::into_raw(data))
}

pub unsafe fn cleanup(ptr: *mut u8) -> Box<dyn Any + Send> {
    Box::from_raw(ptr)
}

