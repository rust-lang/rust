// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Zmir-opt-level=2

pub enum Enum {
    A,
    B,
}

trait SliceIndex {
    type Output;
    fn get(&self) -> &Self::Output;
}

impl SliceIndex for usize {
    type Output = Enum;
    #[inline(never)]
    fn get(&self) -> &Enum {
        &Enum::A
    }
}

#[inline(always)]
fn index<T: SliceIndex>(t: &T) -> &T::Output {
    t.get()
}

fn main() {
    match *index(&0) { Enum::A => true, _ => false };
}
