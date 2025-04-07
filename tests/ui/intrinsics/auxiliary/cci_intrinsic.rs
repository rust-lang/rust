#![feature(intrinsics)]

pub mod rusti {
    #[rustc_intrinsic]
    pub unsafe fn atomic_xchg_seqcst<T>(dst: *mut T, src: T) -> T;
}

#[inline(always)]
pub fn atomic_xchg_seqcst(dst: *mut isize, src: isize) -> isize {
    unsafe { rusti::atomic_xchg_seqcst(dst, src) }
}
