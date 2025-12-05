#![feature(intrinsics, rustc_attrs)]

#[rustc_intrinsic]
#[rustc_nounwind]
#[rustc_do_not_const_check]
#[inline]
pub const fn ptr_guaranteed_cmp<T>(ptr: *const T, other: *const T) -> u8 {
    (ptr == other) as u8
}

fn main() {
    ptr_guaranteed_cmp::<()>(std::ptr::null(), std::ptr::null());
    //~^ ERROR: can only use intrinsic fallback bodies that exactly reflect the specification
}
