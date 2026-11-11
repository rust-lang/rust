use crate::alloc::Layout;

#[inline]
pub unsafe fn alloc(layout: Layout) -> *mut u8 {
    moto_rt::alloc::alloc(layout)
}

#[inline]
pub unsafe fn alloc_zeroed(layout: Layout) -> *mut u8 {
    moto_rt::alloc::alloc_zeroed(layout)
}

pub use moto_rt::alloc::{dealloc, realloc};
