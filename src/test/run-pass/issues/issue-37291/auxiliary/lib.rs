// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "lib"]

use std::ops::Mul;

pub trait A {}
pub trait B {
    type AT: A;
}
pub trait C {
    type BT: B;
}

pub struct AV;
impl A for AV {}

pub struct BV;
impl B for BV {
    type AT = AV;
}

pub struct CV;
impl C for CV {
    type BT = BV;
}

pub struct WrapperB<T>(pub T);
pub struct WrapperC<T>(pub T);

impl<C1> Mul<WrapperB<<C1::BT as B>::AT>> for WrapperC<C1>
    where C1: C
{
    type Output = u8;
    fn mul(self, _: WrapperB<<C1::BT as B>::AT>) -> Self::Output {
        loop {}
    }
}
impl<C1> Mul<WrapperC<C1>> for WrapperC<C1> {
    type Output = u8;
    fn mul(self, _: WrapperC<C1>) -> Self::Output {
        loop {}
    }
}
