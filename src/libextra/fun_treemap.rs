// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * A functional key,value store that works on anything.
 *
 * This works using a binary search tree. In the first version, it's a
 * very naive algorithm, but it will probably be updated to be a
 * red-black tree or something else.
 *
 * This is copied and modified from treemap right now. It's missing a lot
 * of features.
 */

use core::prelude::*;

use core::cmp::{Eq, Ord};
use core::option::{Some, None};

pub type Treemap<K, V> = @TreeNode<K, V>;

enum TreeNode<K, V> {
    Empty,
    Node(@K, @V, @TreeNode<K, V>, @TreeNode<K, V>)
}

/// Create a treemap
pub fn init<K, V>() -> Treemap<K, V> { @Empty }

/// Insert a value into the map
pub fn insert<K:Copy + Eq + Ord,V:Copy>(m: Treemap<K, V>, k: K, v: V) -> Treemap<K, V> {
    @match m {
        @Empty => Node(@k, @v, @Empty, @Empty),
        @Node(@copy kk, vv, left, right) => cond!(
            (k <  kk) { Node(@kk, vv, insert(left, k, v), right) }
            (k == kk) { Node(@kk, @v, left, right)               }
            _         { Node(@kk, vv, left, insert(right, k, v)) }
        )
    }
}

/// Find a value based on the key
pub fn find<K:Eq + Ord,V:Copy>(m: Treemap<K, V>, k: K) -> Option<V> {
    match *m {
        Empty => None,
        Node(@ref kk, @copy v, left, right) => cond!(
            (k == *kk) { Some(v)        }
            (k <  *kk) { find(left, k)  }
            _          { find(right, k) }
        )
    }
}

/// Visit all pairs in the map in order.
pub fn traverse<K, V: Copy>(m: Treemap<K, V>, f: &fn(&K, &V)) {
    match *m {
        Empty => (),
        // Previously, this had what looked like redundant
        // matches to me, so I changed it. but that may be a
        // de-optimization -- tjc
        Node(@ref k, @ref v, left, right) => {
            traverse(left, f);
            f(k, v);
            traverse(right, f);
        }
    }
}
