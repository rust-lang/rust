//@ build-pass
#![allow(internal_features)]
#![feature(rustc_attrs)]

use std::future::{Ready, ready};
use std::ops::{AsyncFn, AsyncFnMut, AsyncFnOnce};

#[rustc_force_inline]
fn f() -> u8 {
    3
}

#[rustc_force_inline]
fn f_async() -> Ready<u8> {
    ready(3)
}

fn call_fn_once<T: FnOnce() -> u8>(t: T) -> u8 {
    t()
}

fn call_fn_mut<T: FnMut() -> u8>(t: &mut T) -> u8 {
    t()
}

fn call_fn<T: Fn() -> u8>(t: &T) -> u8 {
    t()
}

fn call_async_fn_once<T: AsyncFnOnce() -> u8>(t: T) {
    let _ = t();
}

fn call_async_fn_mut<T: AsyncFnMut() -> u8>(t: &mut T) {
    let _ = t();
}

fn call_async_fn<T: AsyncFn() -> u8>(t: &T) {
    let _ = t();
}

fn main() {
    assert_eq!(call_fn_once(f), 3);
    assert_eq!(call_fn_mut(&mut f), 3);
    assert_eq!(call_fn(&f), 3);
    call_async_fn_once(f_async);
    call_async_fn_mut(&mut f_async);
    call_async_fn(&f_async);
}
