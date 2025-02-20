use core::alloc::{AllocError, Allocator, Layout};
use core::mem::{self, zeroed};
use core::ptr::NonNull;

use libc::{c_int, c_void, off_t};

pub type MMap = unsafe extern "C" fn(*mut c_void, usize, c_int, c_int, c_int, i64) -> *mut c_void;
pub type MUnmap = unsafe extern "C" fn(*mut c_void, usize) -> c_int;
pub type Malloc = unsafe extern "C" fn(usize) -> *mut c_void;
pub type Free = unsafe extern "C" fn(*mut c_void);

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct BsanAllocator {
    malloc: Malloc,
    free: Free,
    mmap: MMap,
    munmap: MUnmap,
}

unsafe impl Send for BsanAllocator {}
unsafe impl Sync for BsanAllocator {}

unsafe impl Allocator for BsanAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            match layout.size() {
                0 => Ok(NonNull::slice_from_raw_parts(layout.dangling(), 0)),
                // SAFETY: `layout` is non-zero in size,
                size => unsafe {
                    let raw_ptr: *mut u8 = mem::transmute((self.malloc)(layout.size()));
                    let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
                    Ok(NonNull::slice_from_raw_parts(ptr, size))
                },
            }
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        (self.free)(mem::transmute(ptr.as_ptr()))
    }
}
