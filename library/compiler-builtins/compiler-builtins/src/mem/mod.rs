// Trying to satisfy clippy here is hopeless
#![allow(clippy::style)]
// FIXME(e2024): this eventually needs to be removed.
#![allow(unsafe_op_in_unsafe_fn)]

// memcpy/memmove/memset have optimized implementations on some architectures
#[cfg_attr(all(feature = "arch", target_arch = "x86_64"), path = "x86_64.rs")]
mod impls;

#[cfg(any(windows, target_os = "uefi"))]
#[expect(non_camel_case_types)]
pub type wchar_t = u16;
#[cfg(not(any(windows, target_os = "uefi")))]
#[expect(non_camel_case_types)]
pub type wchar_t = i32;

intrinsics! {
    #[mem_builtin]
    pub unsafe extern "C" fn memcpy(
        dest: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        n: usize
    ) -> *mut core::ffi::c_void {
        impls::copy_forward(dest.cast(), src.cast(), n);
        dest
    }

    #[mem_builtin]
    pub unsafe extern "C" fn memmove(
        dest: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        n: usize
    ) -> *mut core::ffi::c_void {
        let delta = (dest as usize).wrapping_sub(src as usize);
        if delta >= n {
            // We can copy forwards because either dest is far enough ahead of src,
            // or src is ahead of dest (and delta overflowed).
            impls::copy_forward(dest.cast(), src.cast(), n);
        } else {
            impls::copy_backward(dest.cast(), src.cast(), n);
        }
        dest
    }

    #[mem_builtin]
    pub unsafe extern "C" fn memset(
        s: *mut core::ffi::c_void,
        c: core::ffi::c_int,
        n: usize
    ) -> *mut core::ffi::c_void {
        impls::set_bytes(s.cast(), c as u8, n);
        s
    }

    #[mem_builtin]
    pub unsafe extern "C" fn memcmp(
        s1: *const core::ffi::c_void,
        s2: *const core::ffi::c_void,
        n: usize
    ) -> core::ffi::c_int {
        impls::compare_bytes(s1.cast(), s2.cast(), n)
    }

    #[mem_builtin]
    pub unsafe extern "C" fn bcmp(
        s1: *const core::ffi::c_void,
        s2: *const core::ffi::c_void,
        n: usize
    ) -> core::ffi::c_int {
        memcmp(s1, s2, n)
    }

    #[mem_builtin]
    pub unsafe extern "C" fn strlen(s: *const core::ffi::c_char) -> usize {
        impls::c_string_length(s)
    }

    #[mem_builtin]
    pub unsafe extern "C" fn wcslen(s: *const crate::mem::wchar_t) -> usize {
        let mut s = s;
        let mut n = 0;
        while *s != 0 {
            n += 1;
            s = s.add(1);
        }
        n
    }
}
