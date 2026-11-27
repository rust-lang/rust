#![feature(intrinsics, adt_const_params)]

mod rusti {
    #[rustc_intrinsic]
    pub unsafe fn size_of_val<T: ?Sized>(ptr: *const T) -> usize;
}

// A monomorphic function, inlined cross-crate, referencing an intrinsic.
#[inline(always)]
pub fn size_of_val(val: &[u8]) -> usize {
    unsafe { rusti::size_of_val(val) }
}
