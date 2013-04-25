// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn repeat(x: &uint) -> ~[uint] { ~[*x, *x] }

fn incd_if_even(x: &uint) -> Option<uint> {
    if (*x % 2u) == 0u {Some(*x + 1u)} else {None}
}

pub fn main() {
    assert!((~[1u, 3u]).flat_map_to_vec(repeat) == ~[1u, 1u, 3u, 3u]);
    assert!((~[]).flat_map_to_vec(repeat) == ~[]);
    assert!(old_iter::flat_map_to_vec(&None::<uint>, repeat) == ~[]);
    assert!(old_iter::flat_map_to_vec(&Some(1u), repeat) == ~[1u, 1u]);
    assert!(old_iter::flat_map_to_vec(&Some(2u), repeat) == ~[2u, 2u]);

    assert!((~[1u, 2u, 5u]).flat_map_to_vec(incd_if_even) == ~[3u]);
    assert!((~[]).flat_map_to_vec(incd_if_even) == ~[]);
    assert!(old_iter::flat_map_to_vec(&None::<uint>, incd_if_even) == ~[]);
    assert!(old_iter::flat_map_to_vec(&Some(1u), incd_if_even) == ~[]);
    assert!(old_iter::flat_map_to_vec(&Some(2u), incd_if_even) == ~[3u]);
}
