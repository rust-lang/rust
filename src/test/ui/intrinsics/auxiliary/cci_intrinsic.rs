#![feature(intrinsics)]

pub mod rusti {
    extern "rust-intrinsic" {
        pub fn atomic_xchg<T>(dst: *mut T, src: T) -> T;
    }
}

#[inline(always)]
pub fn atomic_xchg(dst: *mut isize, src: isize) -> isize {
    unsafe {
        rusti::atomic_xchg(dst, src)
    }
}
