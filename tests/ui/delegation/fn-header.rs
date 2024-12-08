//@ check-pass
//@ edition:2018
//@ aux-crate:fn_header_aux=fn-header-aux.rs

#![feature(c_variadic)]
#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![deny(unused_unsafe)]

mod to_reuse {
    pub unsafe fn unsafe_fn() {}
    pub extern "C" fn extern_fn() {}
    pub unsafe extern "C" fn variadic_fn(n: usize, mut args: ...) {}
    pub const fn const_fn() {}
    pub async fn async_fn() {}
}

reuse to_reuse::unsafe_fn;
reuse to_reuse::extern_fn;
reuse to_reuse::variadic_fn;
reuse to_reuse::const_fn;
reuse to_reuse::async_fn;

reuse fn_header_aux::unsafe_fn_extern;
reuse fn_header_aux::extern_fn_extern;
reuse fn_header_aux::variadic_fn_extern;
reuse fn_header_aux::const_fn_extern;
reuse fn_header_aux::async_fn_extern;

const fn const_check() {
    const_fn();
    const_fn_extern();
}

async fn async_check() {
    async_fn().await;
    async_fn_extern().await;
}

fn main() {
    unsafe {
        unsafe_fn();
        unsafe_fn_extern();
    }
    extern_fn();
    extern_fn_extern();
    let _: extern "C" fn() = extern_fn;
    let _: extern "C" fn() = extern_fn_extern;
    unsafe {
        variadic_fn(0);
        variadic_fn(0, 1);
        variadic_fn_extern(0);
        variadic_fn_extern(0, 1);
    }
    let _: unsafe extern "C" fn(usize, ...) = variadic_fn;
    let _: unsafe extern "C" fn(usize, ...) = variadic_fn_extern;
}
