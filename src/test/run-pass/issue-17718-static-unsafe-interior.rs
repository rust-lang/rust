// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::kinds::marker;
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

static STATIC1: UnsafeEnum<int> = UnsafeEnum::VariantSafe;

static STATIC2: MyUnsafePack<int> = MyUnsafePack(UnsafeCell { value: 1 });
const CONST: MyUnsafePack<int> = MyUnsafePack(UnsafeCell { value: 1 });
static STATIC3: MyUnsafe<int> = MyUnsafe{value: CONST};

static STATIC4: &'static MyUnsafePack<int> = &STATIC2;

struct Wrap<T> {
    value: T
}

unsafe impl<T: Send> Sync for Wrap<T> {}

static UNSAFE: MyUnsafePack<int> = MyUnsafePack(UnsafeCell{value: 2});
static WRAPPED_UNSAFE: Wrap<&'static MyUnsafePack<int>> = Wrap { value: &UNSAFE };

fn main() {
    let a = &STATIC1;

    STATIC3.forbidden()
}
