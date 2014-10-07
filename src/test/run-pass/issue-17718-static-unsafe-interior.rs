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

struct MyUnsafe<T> {
    value: UnsafeCell<T>
}

impl<T> MyUnsafe<T> {
    fn forbidden(&self) {}
}

enum UnsafeEnum<T> {
    VariantSafe,
    VariantUnsafe(UnsafeCell<T>)
}

static STATIC1: UnsafeEnum<int> = VariantSafe;

static STATIC2: UnsafeCell<int> = UnsafeCell { value: 1 };
const CONST: UnsafeCell<int> = UnsafeCell { value: 1 };
static STATIC3: MyUnsafe<int> = MyUnsafe{value: CONST};

static STATIC4: &'static UnsafeCell<int> = &STATIC2;

struct Wrap<T> {
    value: T
}

static UNSAFE: UnsafeCell<int> = UnsafeCell{value: 1};
static WRAPPED_UNSAFE: Wrap<&'static UnsafeCell<int>> = Wrap { value: &UNSAFE };

fn main() {
    let a = &STATIC1;

    STATIC3.forbidden()
}


