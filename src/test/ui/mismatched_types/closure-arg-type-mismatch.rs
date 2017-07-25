// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
        let a = [(1u32,2u32)];
        let b = a.iter().map(|x: (u32, u32)| 45);
        let d1 = a.iter().map(|x: &(u16,u16)| 45);
        let d2 = a.iter().map(|x: (u16,u16)| 45);
        foo(|y: isize| ());
}

fn foo<F>(m: F) where F: ::std::ops::Fn(usize) {}