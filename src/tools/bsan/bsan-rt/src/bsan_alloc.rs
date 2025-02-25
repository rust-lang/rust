use core::alloc::{AllocError, Allocator, GlobalAlloc, Layout};
use core::mem::{self, zeroed};
use core::ptr::NonNull;

use libc::{c_int, c_void, free, malloc, off_t};

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

#[cfg(test)]
pub static TEST_ALLOC: BsanAllocator = BsanAllocator {
    malloc: libc::malloc,
    free: libc::free,
    mmap: libc::mmap,
    munmap: libc::munmap,
};

/// We need to declare a global allocator to be able to use `alloc` in a `#[no_std]`
/// crate. Anything other than the `BsanAllocator` object will clash with the interceptors,
/// so we use a placeholder that panics when it is used.
#[cfg(not(test))]
mod global_alloc {
    use core::alloc::{GlobalAlloc, Layout};

    #[derive(Default)]
    struct DummyAllocator;

    unsafe impl GlobalAlloc for DummyAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            panic!()
        }
        unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
            panic!()
        }
    }

    #[global_allocator]
    static GLOBAL_ALLOCATOR: DummyAllocator = DummyAllocator;
}
