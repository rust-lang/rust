//@ check-pass
//@ edition:2018

#![feature(c_variadic)]
#![feature(fn_delegation)]
#![allow(incomplete_features)]

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

const fn const_check() {
    const_fn();
}

async fn async_check() {
    async_fn();
}

fn main() {
    unsafe {
        unsafe_fn();
    }
    extern_fn();
    let _: extern "C" fn() = extern_fn;
    unsafe {
        variadic_fn(0);
        variadic_fn(0, 1);
    }
    let _: unsafe extern "C" fn(usize, ...) = variadic_fn;
}
