// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
type IMap<K: Copy, V: Copy> = ~[(K, V)];

trait ImmutableMap<K: Copy, V: Copy>
{
    pure fn contains_key(key: K) -> bool;
}

impl<K: Copy, V: Copy> IMap<K, V> : ImmutableMap<K, V>
{
    pure fn contains_key(key: K) -> bool
    {
        vec::find(self, |e| {e.first() == key}).is_some()
    }
}

fn main() {}