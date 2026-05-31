#[cfg(all(target_os = "windows", feature = "mimalloc"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(all(target_os = "windows", feature = "mimalloc"))]
pub mod fns {
    use std::os::raw::{c_char, c_int, c_void};

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn calloc(items: usize, size: usize) -> *mut c_void {
        unsafe { libmimalloc_sys::mi_calloc(items, size) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn posix_memalign(
        ptr: *mut *mut c_void,
        align: usize,
        size: usize,
    ) -> c_int {
        unsafe { libmimalloc_sys::mi_posix_memalign(ptr, align, size) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn strndup(s: *const c_char, n: usize) -> *mut c_char {
        unsafe { libmimalloc_sys::mi_strndup(s, n) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn strdup(s: *const c_char) -> *mut c_char {
        unsafe { libmimalloc_sys::mi_strdup(s) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn aligned_alloc(align: usize, size: usize) -> *mut c_void {
        unsafe { libmimalloc_sys::mi_aligned_alloc(align, size) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn malloc(size: usize) -> *mut c_void {
        unsafe { libmimalloc_sys::mi_malloc(size) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn realloc(ptr: *mut c_void, size: usize) -> *mut c_void {
        unsafe { libmimalloc_sys::mi_realloc(ptr, size) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn free(ptr: *mut c_void) {
        unsafe {
            libmimalloc_sys::mi_free(ptr);
        }
    }
}
