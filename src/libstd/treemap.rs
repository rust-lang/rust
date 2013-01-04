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
 * A key,value store that works on anything.
 *
 * This works using a binary search tree. In the first version, it's a
 * very naive algorithm, but it will probably be updated to be a
 * red-black tree or something else.
 */
#[forbid(deprecated_mode)];

use core::cmp::{Eq, Ord};
use core::option::{Option, Some, None};
use core::prelude::*;

pub type TreeMap<K: Copy Eq Ord, V: Copy> = @mut TreeEdge<K, V>;

type TreeEdge<K: Copy Eq Ord, V: Copy> = Option<@TreeNode<K, V>>;

struct TreeNode<K: Copy Eq Ord, V: Copy> {
    key: K,
    mut value: V,
    mut left: TreeEdge<K, V>,
    mut right: TreeEdge<K, V>
}

/// Create a treemap
pub fn TreeMap<K: Copy Eq Ord, V: Copy>() -> TreeMap<K, V> { @mut None }

/// Insert a value into the map
pub fn insert<K: Copy Eq Ord, V: Copy>(m: &mut TreeEdge<K, V>, k: K, v: V) {
    match copy *m {
      None => {
        *m = Some(@TreeNode {key: k,
                             mut value: v,
                             mut left: None,
                             mut right: None});
        return;
      }
      Some(node) => {
        if k == node.key {
            node.value = v;
        } else if k < node.key {
            insert(&mut node.left, k, v);
        } else {
            insert(&mut node.right, k, v);
        }
      }
    };
}

/// Find a value based on the key
pub fn find<K: Copy Eq Ord, V: Copy>(m: &const TreeEdge<K, V>, k: K)
                              -> Option<V> {
    match copy *m {
      None => None,

      // FIXME (#2808): was that an optimization?
      Some(node) => {
        if k == node.key {
            Some(node.value)
        } else if k < node.key {
            find(&const node.left, k)
        } else {
            find(&const node.right, k)
        }
      }
    }
}

/// Visit all pairs in the map in order.
pub fn traverse<K: Copy Eq Ord, V: Copy>(m: &const TreeEdge<K, V>,
                                         f: fn((&K), (&V))) {
    match copy *m {
      None => (),
      Some(node) => {
        traverse(&const node.left, f);
        // copy of value is req'd as f() requires an immutable ptr
        f(&node.key, &copy node.value);
        traverse(&const node.right, f);
      }
    }
}

/// Compare two treemaps and return true iff
/// they contain same keys and values
pub fn equals<K: Copy Eq Ord, V: Copy Eq>(t1: &const TreeEdge<K, V>,
                                          t2: &const TreeEdge<K, V>)
                                        -> bool {
    let mut v1 = ~[];
    let mut v2 = ~[];
    traverse(t1, |k,v| { v1.push((copy *k, copy *v)) });
    traverse(t2, |k,v| { v2.push((copy *k, copy *v)) });
    return v1 == v2;
}


#[cfg(test)]
mod tests {
    #[legacy_exports];

    use treemap::*;

    use core::option::{None, Option, Some};
    use core::str;

    #[test]
    fn init_treemap() { let _m = TreeMap::<int, int>(); }

    #[test]
    fn insert_one() { let m = TreeMap(); insert(m, 1, 2); }

    #[test]
    fn insert_two() { let m = TreeMap(); insert(m, 1, 2); insert(m, 3, 4); }

    #[test]
    fn insert_find() {
        let m = TreeMap();
        insert(m, 1, 2);
        assert (find(m, 1) == Some(2));
    }

    #[test]
    fn find_empty() {
        let m = TreeMap::<int, int>(); assert (find(m, 1) == None);
    }

    #[test]
    fn find_not_found() {
        let m = TreeMap();
        insert(m, 1, 2);
        assert (find(m, 2) == None);
    }

    #[test]
    fn traverse_in_order() {
        let m = TreeMap();
        insert(m, 3, ());
        insert(m, 0, ());
        insert(m, 4, ());
        insert(m, 2, ());
        insert(m, 1, ());

        let n = @mut 0;
        fn t(n: @mut int, k: int, _v: ()) {
            assert (*n == k); *n += 1;
        }
        traverse(m, |x,y| t(n, *x, *y));
    }

    #[test]
    fn equality() {
        let m1 = TreeMap();
        insert(m1, 3, ());
        insert(m1, 0, ());
        insert(m1, 4, ());
        insert(m1, 2, ());
        insert(m1, 1, ());
        let m2 = TreeMap();
        insert(m2, 2, ());
        insert(m2, 1, ());
        insert(m2, 3, ());
        insert(m2, 0, ());
        insert(m2, 4, ());

        assert equals(m1, m2);

        let m3 = TreeMap();
        assert !equals(m1,m3);

    }

    #[test]
    fn u8_map() {
        let m = TreeMap();

        let k1 = str::to_bytes(~"foo");
        let k2 = str::to_bytes(~"bar");

        insert(m, k1, ~"foo");
        insert(m, k2, ~"bar");

        assert (find(m, k2) == Some(~"bar"));
        assert (find(m, k1) == Some(~"foo"));
    }
}
