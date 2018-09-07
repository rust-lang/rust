// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type="rlib"]

#[inline(never)]
pub fn foo<T>() {
    let _: Box<SomeTrait> = Box::new(SomeTraitImpl);
}

pub fn bar() {
    SomeTraitImpl.bar();
}

mod submod {
    pub trait SomeTrait {
        fn bar(&self) {
            panic!("NO")
        }
    }
}

use self::submod::SomeTrait;

pub struct SomeTraitImpl;
impl SomeTrait for SomeTraitImpl {}
