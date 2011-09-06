/*
A key,value store that works on anything.

This works using a binary search tree. In the first version, it's a
very naive algorithm, but it will probably be updated to be a
red-black tree or something else.

*/

import option::some;
import option::none;
import option = option::t;

export treemap;
export init;
export insert;
export find;
export traverse;

tag tree_node<@K, @V> { empty; node(@K, @V, treemap<K, V>, treemap<K, V>); }

type treemap<@K, @V> = @mutable tree_node<K, V>;

fn init<@K, @V>() -> treemap<K, V> { @mutable empty }

fn insert<@K, @V>(m: &treemap<K, V>, k: &K, v: &V) {
    alt m {
      @empty. { *m = node(@k, @v, @mutable empty, @mutable empty); }
      @node(@kk, _, _, _) {

        // We have to name left and right individually, because
        // otherwise the alias checker complains.
        if k < kk {
            alt m { @node(_, _, left, _) { insert(left, k, v); } }
        } else { alt m { @node(_, _, _, right) { insert(right, k, v); } } }
      }
    }
}

fn find<@K, @V>(m: &treemap<K, V>, k: &K) -> option<V> {
    alt *m {
      empty. { none }
      node(@kk, @v, _, _) {
        if k == kk {
            some(v)
        } else if k < kk {
            // Again, ugliness to unpack left and right individually.
            alt *m { node(_, _, left, _) { find(left, k) } }
        } else { alt *m { node(_, _, _, right) { find(right, k) } } }
      }
    }
}

// Performs an in-order traversal
fn traverse<@K, @V>(m: &treemap<K, V>, f: fn(&K, &V)) {
    alt *m {
      empty. { }
      node(k, v, _, _) {
        let k1 = k, v1 = v;
        alt *m { node(_, _, left, _) { traverse(left, f); } }
        f(*k1, *v1);
        alt *m { node(_, _, _, right) { traverse(right, f); } }
      }
    }
}
