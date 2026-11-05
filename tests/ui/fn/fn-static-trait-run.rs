#![feature(fn_static)]
//@ run-pass

use std::ops::FnStatic;

fn bar(a: u64) -> u64 {
    a
}

fn call_me<F: FnStatic + Fn(u64) -> u64>(f: F, a: u64) -> u64 {
    assert_eq!(f.addr(), F::INSTANCE.addr());
    F::INSTANCE(a)
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
}
