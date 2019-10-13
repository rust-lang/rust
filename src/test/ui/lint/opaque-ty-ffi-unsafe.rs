#![feature(type_alias_impl_trait)]

#![deny(improper_ctypes)]

type A = impl Fn();

pub fn ret_closure() -> A {
    || {}
}

extern "C" {
    pub fn a(_: A);
    //~^ ERROR `extern` block uses type `A`, which is not FFI-safe
}

fn main() {}
