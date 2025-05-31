#![feature(intrinsics, adt_const_params)]

pub mod rusti {
    use std::marker::ConstParamTy;

    #[derive(Debug, ConstParamTy, PartialEq, Eq)]
    pub enum AtomicOrdering {
        // These values must match the compiler's `AtomicOrdering` defined in
        // `rustc_middle/src/ty/consts/int.rs`!
        Relaxed = 0,
        Release = 1,
        Acquire = 2,
        AcqRel = 3,
        SeqCst = 4,
    }

    #[rustc_intrinsic]
    pub unsafe fn atomic_xchg<T, const ORD: AtomicOrdering>(dst: *mut T, src: T) -> T;
}

#[inline(always)]
pub fn atomic_xchg_seqcst(dst: *mut isize, src: isize) -> isize {
    unsafe { rusti::atomic_xchg::<_, { rusti::AtomicOrdering::SeqCst }>(dst, src) }
}
