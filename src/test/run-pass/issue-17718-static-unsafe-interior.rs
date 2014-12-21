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
use std::cell::{UnsafeCell, RacyCell};

struct MyUnsafe<T> {
    value: RacyCell<T>
}

impl<T> MyUnsafe<T> {
    fn forbidden(&self) {}
}

enum UnsafeEnum<T> {
    VariantSafe,
    VariantUnsafe(RacyCell<T>)
}

static STATIC1: UnsafeEnum<int> = UnsafeEnum::VariantSafe;

static STATIC2: RacyCell<int> = RacyCell(UnsafeCell { value: 1 });
const CONST: RacyCell<int> = RacyCell(UnsafeCell { value: 1 });
static STATIC3: MyUnsafe<int> = MyUnsafe{value: CONST};

static STATIC4: &'static RacyCell<int> = &STATIC2;

struct Wrap<T> {
    value: T
}

unsafe impl<T: Send> Sync for Wrap<T> {}

static UNSAFE: RacyCell<int> = RacyCell(UnsafeCell{value: 1});
static WRAPPED_UNSAFE: Wrap<&'static RacyCell<int>> = Wrap { value: &UNSAFE };

fn main() {
    let a = &STATIC1;

    STATIC3.forbidden()
}


