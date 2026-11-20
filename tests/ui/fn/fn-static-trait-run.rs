#![feature(fn_static)]
//@ run-pass

use std::ops::{FnAddr, FnStatic};

fn bar(a: u64) -> u64 {
    a
}

unsafe fn foo(a: u64) -> u64 {
    a
}

fn call_me<F: FnStatic + Fn(u64) -> u64>(f: F, a: u64) -> u64 {
    assert_eq!(f.addr(), F::INSTANCE.addr());
    assert_eq!(f.addr(), f.as_fn().addr());
    F::INSTANCE(a)
}

fn call_me_unsafe<F: FnStatic<FnPtr = unsafe fn(u64) -> u64>>(f: F, a: u64) -> u64 {
    assert_eq!(f.addr(), F::INSTANCE.addr());
    unsafe { F::INSTANCE.as_fn()(a) }
}

fn get_wrapper<F: FnStatic + Fn(u64) -> u64>(_: F) -> fn(u64) -> u64 {
    fn wrapper<F: FnStatic + Fn(u64) -> u64>(a: u64) -> u64 {
        F::INSTANCE(a + 1)
    }
    wrapper::<F>
}

fn main() {
    assert_eq!(call_me(bar, 42), 42);
    assert_eq!(get_wrapper(bar)(69), 70);
    assert_eq!(call_me(|x: u64| x * 2, 10), 20);
    assert_eq!(call_me_unsafe(foo, 42), 42);
}
