//@ edition:2018

#![feature(c_variadic)]

pub unsafe fn unsafe_fn_extern() {}
pub extern "C" fn extern_fn_extern() {}
pub unsafe extern "C" fn variadic_fn_extern(n: usize, mut args: ...) {}
pub const fn const_fn_extern() {}
pub async fn async_fn_extern() {}
