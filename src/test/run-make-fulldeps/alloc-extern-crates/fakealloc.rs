#![crate_type = "rlib"]
#![no_std]

#[inline]
pub unsafe fn allocate(_size: usize, _align: usize) -> *mut u8 {
    core::ptr::null_mut()
}

#[inline]
pub unsafe fn deallocate(_ptr: *mut u8, _old_size: usize, _align: usize) { }

#[inline]
pub unsafe fn reallocate(_ptr: *mut u8, _old_size: usize, _size: usize, _align: usize) -> *mut u8 {
    core::ptr::null_mut()
}

#[inline]
pub unsafe fn reallocate_inplace(_ptr: *mut u8, old_size: usize, _size: usize,
                                    _align: usize) -> usize { old_size }

#[inline]
pub fn usable_size(size: usize, _align: usize) -> usize { size }

#[inline]
pub fn stats_print() { }
