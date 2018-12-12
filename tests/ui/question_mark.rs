// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn some_func(a: Option<u32>) -> Option<u32> {
    if a.is_none() {
        return None;
    }

    a
}

pub enum SeemsOption<T> {
    Some(T),
    None,
}

impl<T> SeemsOption<T> {
    pub fn is_none(&self) -> bool {
        match *self {
            SeemsOption::None => true,
            SeemsOption::Some(_) => false,
        }
    }
}

fn returns_something_similar_to_option(a: SeemsOption<u32>) -> SeemsOption<u32> {
    if a.is_none() {
        return SeemsOption::None;
    }

    a
}

pub struct SomeStruct {
    pub opt: Option<u32>,
}

impl SomeStruct {
    pub fn func(&self) -> Option<u32> {
        if (self.opt).is_none() {
            return None;
        }

        self.opt
    }
}

fn main() {
    some_func(Some(42));
    some_func(None);

    let some_struct = SomeStruct { opt: Some(54) };
    some_struct.func();

    let so = SeemsOption::Some(45);
    returns_something_similar_to_option(so);
}
