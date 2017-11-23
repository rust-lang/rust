// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(default_type_parameter_fallback)]

struct HeshMep<K, V, S=String>(Vec<K>, Vec<V>, S);

impl<V, K=usize, S:Default=String> HeshMep<K, V, S> {
    fn new() -> HeshMep<K, V, S> {
        HeshMep(Vec::new(), Vec::new(), S::default())
    }
}

type IntMap<K> = HeshMep<K, usize>;

fn main() {
    let _ = IntMap::new();
}
