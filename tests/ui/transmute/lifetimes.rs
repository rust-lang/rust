//@ check-pass
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

use std::ptr::NonNull;

struct Foo<'a, T>(&'a (), NonNull<T>);

fn foo<'a, 'b, T>(x: Foo<'a, T>) -> Foo<'b, T> {
    unsafe { std::mem::transmute(x) }
}

struct Bar<'a, T>(&'a T);

fn bar<'a, 'b, T>(x: Bar<'a, T>) -> Bar<'b, T> {
    unsafe { std::mem::transmute(x) }
}

struct Boo<'a, T>(&'a T, u32);

fn boo<'a, 'b, T>(x: Boo<'a, T>) -> Boo<'b, T> {
    unsafe { std::mem::transmute(x) }
}

fn main() {}
