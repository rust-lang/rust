//@ aux-crate:fn_header_aux=fn-header-aux.rs
//@ ignore-backends: gcc

#![feature(c_variadic)]
#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod to_reuse {
    pub unsafe extern "C" fn variadic_fn(n: usize, mut args: ...) {}
}

reuse to_reuse::variadic_fn;
//~^ ERROR delegation to C-variadic functions is not allowed
reuse fn_header_aux::variadic_fn_extern;
//~^ ERROR delegation to C-variadic functions is not allowed

fn main() {
    unsafe {
        variadic_fn(0);
        variadic_fn(0, 1);
        variadic_fn_extern(0);
        variadic_fn_extern(0, 1);
    }
    let _: unsafe extern "C" fn(usize, ...) = variadic_fn;
    let _: unsafe extern "C" fn(usize, ...) = variadic_fn_extern;
}
