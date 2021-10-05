const fn bad_const_fn_deref_raw(x: *mut usize) -> &'static usize { unsafe { &*x } }
//~^ dereferencing raw mutable pointers in constant functions

const unsafe fn bad_const_unsafe_deref_raw(x: *mut usize) -> usize { *x }
//~^ dereferencing raw mutable pointers in constant functions

const unsafe fn bad_const_unsafe_deref_raw_ref(x: *mut usize) -> &'static usize { &*x }
//~^ dereferencing raw mutable pointers in constant functions

fn main() {}
