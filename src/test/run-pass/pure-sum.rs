// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that pure functions can modify local state.

pure fn sums_to(v: ~[int], sum: int) -> bool {
    let mut i = 0u, sum0 = 0;
    while i < v.len() {
        sum0 += v[i];
        i += 1u;
    }
    return sum0 == sum;
}

pure fn sums_to_using_uniq(v: ~[int], sum: int) -> bool {
    let mut i = 0u, sum0 = ~mut 0;
    while i < v.len() {
        *sum0 += v[i];
        i += 1u;
    }
    return *sum0 == sum;
}

pure fn sums_to_using_rec(v: ~[int], sum: int) -> bool {
    let mut i = 0u, sum0 = {f: 0};
    while i < v.len() {
        sum0.f += v[i];
        i += 1u;
    }
    return sum0.f == sum;
}

pure fn sums_to_using_uniq_rec(v: ~[int], sum: int) -> bool {
    let mut i = 0u, sum0 = {f: ~mut 0};
    while i < v.len() {
        *sum0.f += v[i];
        i += 1u;
    }
    return *sum0.f == sum;
}

fn main() {
}