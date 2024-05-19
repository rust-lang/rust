//@ check-pass

use std::ptr::NonNull;

struct Foo<'a, T: ?Sized>(&'a (), NonNull<T>);

fn foo<'a, 'b, T: ?Sized>(x: Foo<'a, T>) -> Foo<'b, T> {
    unsafe { std::mem::transmute(x) }
}

struct Bar<'a, T: ?Sized>(&'a T);

fn bar<'a, 'b, T: ?Sized>(x: Bar<'a, T>) -> Bar<'b, T> {
    unsafe { std::mem::transmute(x) }
}

struct Boo<'a, T: ?Sized>(&'a T, u32);

fn boo<'a, 'b, T: ?Sized>(x: Boo<'a, T>) -> Boo<'b, T> {
    unsafe { std::mem::transmute(x) }
}

fn main() {}
