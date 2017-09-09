// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(drop_types_in_const)]

struct WithDtor;

impl Drop for WithDtor {
    fn drop(&mut self) {}
}

static PROMOTION_FAIL_S: Option<&'static WithDtor> = Some(&WithDtor);
//~^ ERROR statics are not allowed to have destructors
//~| ERROR borrowed value does not live long enoug

const PROMOTION_FAIL_C: Option<&'static WithDtor> = Some(&WithDtor);
//~^ ERROR constants are not allowed to have destructors
//~| ERROR borrowed value does not live long enoug

static EARLY_DROP_S: i32 = (WithDtor, 0).1;
//~^ ERROR statics are not allowed to have destructors

const EARLY_DROP_C: i32 = (WithDtor, 0).1;
//~^ ERROR constants are not allowed to have destructors

fn main () {}
