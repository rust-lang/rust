// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// tests that ctrl's type gets inferred properly
type command<K: Owned, V: Owned> = {key: K, val: V};

fn cache_server<K: Owned, V: Owned>(c: oldcomm::Chan<oldcomm::Chan<command<K, V>>>) {
    let ctrl = oldcomm::Port();
    oldcomm::send(c, oldcomm::Chan(&ctrl));
}
fn main() { }
