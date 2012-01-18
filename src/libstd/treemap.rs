/*
Module: treemap

A key,value store that works on anything.

This works using a binary search tree. In the first version, it's a
very naive algorithm, but it will probably be updated to be a
red-black tree or something else.

*/

import core::option::{some, none};
import option = core::option::t;

export treemap;
export init;
export insert;
export find;
export traverse;

/* Section: Types */

/*
Type: treemap
*/
type treemap<K, V> = @mutable tree_node<K, V>;

/*
Tag: tree_node
*/
tag tree_node<K, V> { empty; node(@K, @V, treemap<K, V>, treemap<K, V>); }

/* Section: Operations */

/*
Function: init

Create a treemap
*/
fn init<K, V>() -> treemap<K, V> { @mutable empty }

/*
Function: insert

Insert a value into the map
*/
fn insert<K: copy, V: copy>(m: treemap<K, V>, k: K, v: V) {
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

/*
Function: find

Find a value based on the key
*/
fn find<K: copy, V: copy>(m: treemap<K, V>, k: K) -> option<V> {
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

/*
Function: traverse

Visit all pairs in the map in order.
*/
fn traverse<K, V>(m: treemap<K, V>, f: block(K, V)) {
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

#[cfg(test)]
mod tests {

    #[test]
    fn init_treemap() { let _m = init::<int, int>(); }

    #[test]
    fn insert_one() { let m = init(); insert(m, 1, 2); }

    #[test]
    fn insert_two() { let m = init(); insert(m, 1, 2); insert(m, 3, 4); }

    #[test]
    fn insert_find() {
        let m = init();
        insert(m, 1, 2);
        assert (find(m, 1) == some(2));
    }

    #[test]
    fn find_empty() {
        let m = init::<int, int>(); assert (find(m, 1) == none);
    }

    #[test]
    fn find_not_found() {
        let m = init();
        insert(m, 1, 2);
        assert (find(m, 2) == none);
    }

    #[test]
    fn traverse_in_order() {
        let m = init();
        insert(m, 3, ());
        insert(m, 0, ());
        insert(m, 4, ());
        insert(m, 2, ());
        insert(m, 1, ());

        let n = @mutable 0;
        fn t(n: @mutable int, &&k: int, &&_v: ()) {
            assert (*n == k); *n += 1;
        }
        traverse(m, bind t(n, _, _));
    }

    #[test]
    fn u8_map() {
        let m = init();

        let k1 = str::bytes("foo");
        let k2 = str::bytes("bar");

        insert(m, k1, "foo");
        insert(m, k2, "bar");

        assert (find(m, k2) == some("bar"));
        assert (find(m, k1) == some("foo"));
    }
}