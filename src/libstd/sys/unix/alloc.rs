use crate::ptr;
use crate::sys_common::alloc::{MIN_ALIGN, realloc_fallback};
use crate::alloc::{GlobalAlloc, Layout, System};

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
            libc::malloc(layout.size()) as *mut u8
        } else {
            #[cfg(target_os = "macos")]
            {
                if layout.align() > (1 << 31) {
                    return ptr::null_mut()
                }
            }
            aligned_malloc(&layout)
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
            libc::calloc(layout.size(), 1) as *mut u8
        } else {
            let ptr = self.alloc(layout.clone());
            if !ptr.is_null() {
                ptr::write_bytes(ptr, 0, layout.size());
            }
            ptr
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        libc::free(ptr as *mut libc::c_void)
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.align() <= MIN_ALIGN && layout.align() <= new_size {
            libc::realloc(ptr as *mut libc::c_void, new_size) as *mut u8
        } else {
            realloc_fallback(self, ptr, layout, new_size)
        }
    }
}

#[cfg(any(target_os = "android",
          target_os = "hermit",
          target_os = "redox",
          target_os = "solaris"))]
#[inline]
unsafe fn aligned_malloc(layout: &Layout) -> *mut u8 {
    // On android we currently target API level 9 which unfortunately
    // doesn't have the `posix_memalign` API used below. Instead we use
    // `memalign`, but this unfortunately has the property on some systems
    // where the memory returned cannot be deallocated by `free`!
    //
    // Upon closer inspection, however, this appears to work just fine with
    // Android, so for this platform we should be fine to call `memalign`
    // (which is present in API level 9). Some helpful references could
    // possibly be chromium using memalign [1], attempts at documenting that
    // memalign + free is ok [2] [3], or the current source of chromium
    // which still uses memalign on android [4].
    //
    // [1]: https://codereview.chromium.org/10796020/
    // [2]: https://code.google.com/p/android/issues/detail?id=35391
    // [3]: https://bugs.chromium.org/p/chromium/issues/detail?id=138579
    // [4]: https://chromium.googlesource.com/chromium/src/base/+/master/
    //                                       /memory/aligned_memory.cc
    libc::memalign(layout.align(), layout.size()) as *mut u8
}

#[cfg(not(any(target_os = "android",
              target_os = "hermit",
              target_os = "redox",
              target_os = "solaris")))]
#[inline]
unsafe fn aligned_malloc(layout: &Layout) -> *mut u8 {
    let mut out = ptr::null_mut();
    let ret = libc::posix_memalign(&mut out, layout.align(), layout.size());
    if ret != 0 {
        ptr::null_mut()
    } else {
        out as *mut u8
    }
}
