#![allow(missing_docs)]
use core::ptr::NonNull;
use core::ffi::c_void;

#[stable(feature = "alloc_module", since = "1.28.0")]
#[doc(inline)]
use core::alloc::*;

#[derive(Debug)]
pub struct BoehmAllocator;

#[unstable(feature = "allocator_api", issue = "32838")]
unsafe impl AllocRef for BoehmAllocator {
    fn alloc(&mut self, layout: Layout, _init: AllocInit) -> Result<MemoryBlock, AllocErr> {
        let ptr = unsafe { GC_malloc(layout.size()) } as *mut u8;
        assert!(!ptr.is_null());
        Ok(MemoryBlock {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            size: layout.size(),
        })
    }

    unsafe fn dealloc(&mut self, _: NonNull<u8>, _: Layout) {}
}


#[link(name = "gc")]
extern "C" {
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn GC_gcollect();

    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn GC_malloc(nbytes: usize) -> *mut c_void;

    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn GC_malloc_uncollectable(nbytes: usize) -> *mut c_void;

    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn GC_realloc(old: *mut c_void, new_size: usize) -> *mut c_void;

    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn GC_free(dead: *mut c_void);

    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn GC_register_finalizer(
        ptr: *mut c_void,
        finalizer: unsafe extern "C" fn(*mut c_void, *mut c_void),
        client_data: *mut c_void,
        old_finalizer: *mut extern "C" fn(*mut c_void, *mut c_void),
        old_client_data: *mut *mut c_void,
    );
}
