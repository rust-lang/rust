//@ check-pass

const unsafe fn ret_i32_no_unsafe() -> i32 { 42 }
const unsafe fn ret_null_ptr_no_unsafe<T>() -> *const T { std::ptr::null() }
const unsafe fn ret_null_mut_ptr_no_unsafe<T>() -> *mut T { std::ptr::null_mut() }
const fn no_unsafe() { unsafe {} }

const fn call_unsafe_const_fn() -> i32 {
    unsafe { ret_i32_no_unsafe() }
}
const fn call_unsafe_generic_const_fn() -> *const String {
    unsafe { ret_null_ptr_no_unsafe::<String>() }
}
const fn call_unsafe_generic_cell_const_fn()
    -> *const Vec<std::cell::Cell<u32>>
{
    unsafe { ret_null_mut_ptr_no_unsafe::<Vec<std::cell::Cell<u32>>>() }
}

const unsafe fn call_unsafe_const_unsafe_fn() -> i32 {
    unsafe { ret_i32_no_unsafe() }
}
const unsafe fn call_unsafe_generic_const_unsafe_fn() -> *const String {
    unsafe { ret_null_ptr_no_unsafe::<String>() }
}
const unsafe fn call_unsafe_generic_cell_const_unsafe_fn()
    -> *const Vec<std::cell::Cell<u32>>
{
    unsafe { ret_null_mut_ptr_no_unsafe::<Vec<std::cell::Cell<u32>>>() }
}

const unsafe fn call_unsafe_const_unsafe_fn_immediate() -> i32 {
    ret_i32_no_unsafe()
}
const unsafe fn call_unsafe_generic_const_unsafe_fn_immediate() -> *const String {
    ret_null_ptr_no_unsafe::<String>()
}
const unsafe fn call_unsafe_generic_cell_const_unsafe_fn_immediate()
    -> *const Vec<std::cell::Cell<u32>>
{
    ret_null_mut_ptr_no_unsafe::<Vec<std::cell::Cell<u32>>>()
}

const fn bad_const_fn_deref_raw(x: *mut usize) -> &'static usize { unsafe { &*x } }

const unsafe fn bad_const_unsafe_deref_raw(x: *mut usize) -> usize { *x }

const unsafe fn bad_const_unsafe_deref_raw_ref(x: *mut usize) -> &'static usize { &*x }

const unsafe fn bad_const_unsafe_deref_raw_underscore(x: *mut usize) { let _ = *x; }

fn main() {}
