// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ref: https://github.com/rust-lang/rust/issues/23563#issuecomment-260751672

pub trait LolTo<T> {
    fn convert_to(&self) -> T;
}

pub trait LolInto<T>: Sized {
    fn convert_into(self) -> T;
}

pub trait LolFrom<T> {
    fn from(T) -> Self;
}

impl<'a, T: ?Sized, U> LolInto<U> for &'a T where T: LolTo<U> {
    fn convert_into(self) -> U {
        self.convert_to()
    }
}

impl<T, U> LolFrom<T> for U where T: LolInto<U> {
    fn from(t: T) -> U {
        t.convert_into()
    }
}
