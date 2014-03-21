// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


enum option<T> { some(T), none, }

struct R<T> {v: Vec<option<T>> }

fn f<T>() -> Vec<T> { return Vec::new(); }

pub fn main() { let mut r: R<int> = R {v: Vec::new()}; r.v = f(); }
