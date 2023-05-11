#![feature(explicit_tail_calls)]
#![feature(c_variadic)]

pub unsafe extern "C" fn c_variadic_f(x: u8, mut args: ...) {
    become c_variadic_g(x)
    //~^ error: tail-calls are not allowed in c-variadic functions
    //~| error: c-variadic functions can't be tail-called
}

pub unsafe extern "C" fn c_variadic_g(x: u8, mut args: ...) {
    become normal(x)
    //~^ error: tail-calls are not allowed in c-variadic functions
}

unsafe extern "C" fn normal(x: u8) {
    become c_variadic_f(x)
    //~^ error: c-variadic functions can't be tail-called
}

fn main() {}
