#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

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

use core::cmp::{Eq, Ord};
use option::{Some, None};
use option = option;

export Treemap;
export init;
export insert;
export find;
export traverse;

type Treemap<K, V> = @TreeNode<K, V>;

enum TreeNode<K, V> {
    Empty,
    Node(@K, @V, @TreeNode<K, V>, @TreeNode<K, V>)
}

/// Create a treemap
fn init<K, V>() -> Treemap<K, V> { @Empty }

/// Insert a value into the map
fn insert<K: Copy Eq Ord, V: Copy>(m: Treemap<K, V>, +k: K, +v: V)
  -> Treemap<K, V> {
    @match m {
       @Empty => Node(@k, @v, @Empty, @Empty),
       @Node(@kk, vv, left, right) => {
         if k < kk {
             Node(@kk, vv, insert(left, k, v), right)
         } else if k == kk {
             Node(@kk, @v, left, right)
         } else { Node(@kk, vv, left, insert(right, k, v)) }
       }
     }
}

/// Find a value based on the key
fn find<K: Eq Ord, V: Copy>(m: Treemap<K, V>, +k: K) -> Option<V> {
    match *m {
      Empty => None,
      Node(@kk, @v, left, right) => {
        if k == kk {
            Some(v)
        } else if k < kk { find(left, move k) } else { find(right, move k) }
      }
    }
}

/// Visit all pairs in the map in order.
fn traverse<K, V: Copy>(m: Treemap<K, V>, f: fn(K, V)) {
    match *m {
      Empty => (),
      /*
        Previously, this had what looked like redundant
        matches to me, so I changed it. but that may be a
        de-optimization -- tjc
       */
      Node(@k, @v, left, right) => {
        // copy v to make aliases work out
        let v1 = v;
        traverse(left, f);
        f(k, v1);
        traverse(right, f);
      }
    }
}
