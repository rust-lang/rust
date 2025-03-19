// Trying to satisfy clippy here is hopeless
#![allow(clippy::style)]
// FIXME(e2024): this eventually needs to be removed.
#![allow(unsafe_op_in_unsafe_fn)]

#[allow(warnings)]
#[cfg(target_pointer_width = "16")]
type c_int = i16;
#[allow(warnings)]
#[cfg(not(target_pointer_width = "16"))]
type c_int = i32;

// memcpy/memmove/memset have optimized implementations on some architectures
#[cfg_attr(
    all(not(feature = "no-asm"), target_arch = "x86_64"),
    path = "x86_64.rs"
)]
mod impls;

intrinsics! {
    #[mem_builtin]
    pub unsafe extern "C" fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
        impls::copy_forward(dest, src, n);
        dest
    }

    #[mem_builtin]
    pub unsafe extern "C" fn memmove(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
        let delta = (dest as usize).wrapping_sub(src as usize);
        if delta >= n {
            // We can copy forwards because either dest is far enough ahead of src,
            // or src is ahead of dest (and delta overflowed).
            impls::copy_forward(dest, src, n);
        } else {
            impls::copy_backward(dest, src, n);
        }
        dest
    }

    #[mem_builtin]
    pub unsafe extern "C" fn memset(s: *mut u8, c: crate::mem::c_int, n: usize) -> *mut u8 {
        impls::set_bytes(s, c as u8, n);
        s
    }

    #[mem_builtin]
    pub unsafe extern "C" fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32 {
        impls::compare_bytes(s1, s2, n)
    }

    #[mem_builtin]
    pub unsafe extern "C" fn bcmp(s1: *const u8, s2: *const u8, n: usize) -> i32 {
        memcmp(s1, s2, n)
    }

    #[mem_builtin]
    pub unsafe extern "C" fn strlen(s: *const core::ffi::c_char) -> usize {
        impls::c_string_length(s)
    }
}
