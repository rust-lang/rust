//@ run-pass

#![feature(unsize)]
#![feature(coerce_unsized)]

use std::fmt::Display;
use std::marker::Unsize;
use std::ops::CoerceUnsized;
use std::rc::Weak;

#[repr(transparent)]
struct X<'a, T: ?Sized> {
    f: &'a T,
}

impl<'a, T: ?Sized> Drop for X<'a, T> {
    fn drop(&mut self) {
        panic!()
    }
}

impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<X<'a, U>> for X<'a, T> where
    &'a T: CoerceUnsized<&'a U>
{
}

const Y: X<'static, i32> = X { f: &0 };

fn main() {
    let _: [X<'static, dyn Display>; 0] = [Y; 0];
    coercion_on_weak_in_const();
    coercion_on_weak_as_cast();
}

fn coercion_on_weak_in_const() {
    const X: Weak<i32> = Weak::new();
    const Y: [Weak<dyn Send>; 0] = [X; 0];
    let _ = Y;
}

fn coercion_on_weak_as_cast() {
    const Y: X<'static, i32> = X { f: &0 };
    // What happens in the following code is that
    // a constant is explicitly coerced into
    let _a: [X<'static, dyn Display>; 0] = [Y as X<'static, dyn Display>; 0];
}
