//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::marker;
use std::cell::UnsafeCell;

struct MyUnsafePack<T>(UnsafeCell<T>);

unsafe impl<T: Send> Sync for MyUnsafePack<T> {}

struct MyUnsafe<T> {
    value: MyUnsafePack<T>
}

impl<T> MyUnsafe<T> {
    fn forbidden(&self) {}
}

unsafe impl<T: Send> Sync for MyUnsafe<T> {}

enum UnsafeEnum<T> {
    VariantSafe,
    VariantUnsafe(UnsafeCell<T>)
}

unsafe impl<T: Send> Sync for UnsafeEnum<T> {}

static STATIC1: UnsafeEnum<isize> = UnsafeEnum::VariantSafe;

static STATIC2: MyUnsafePack<isize> = MyUnsafePack(UnsafeCell::new(1));
const CONST: MyUnsafePack<isize> = MyUnsafePack(UnsafeCell::new(1));
static STATIC3: MyUnsafe<isize> = MyUnsafe{value: CONST};

static STATIC4: &'static MyUnsafePack<isize> = &STATIC2;

struct Wrap<T> {
    value: T
}

unsafe impl<T: Send> Sync for Wrap<T> {}

static UNSAFE: MyUnsafePack<isize> = MyUnsafePack(UnsafeCell::new(2));
static WRAPPED_UNSAFE: Wrap<&'static MyUnsafePack<isize>> = Wrap { value: &UNSAFE };

fn main() {
    let a = &STATIC1;

    STATIC3.forbidden()
}
