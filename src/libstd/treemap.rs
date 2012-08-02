/*!
 * A key,value store that works on anything.
 *
 * This works using a binary search tree. In the first version, it's a
 * very naive algorithm, but it will probably be updated to be a
 * red-black tree or something else.
 */

import core::option::{some, none};
import option = core::option;

export treemap;
export insert;
export find;
export traverse;

type treemap<K, V> = @mut tree_edge<K, V>;

type tree_edge<K, V> = option<@tree_node<K, V>>;

enum tree_node<K, V> = {
    key: K,
    mut value: V,
    mut left: tree_edge<K, V>,
    mut right: tree_edge<K, V>
};

/// Create a treemap
fn treemap<K, V>() -> treemap<K, V> { @mut none }

/// Insert a value into the map
fn insert<K: copy, V: copy>(m: &mut tree_edge<K, V>, k: K, v: V) {
    alt copy *m {
      none {
        *m = some(@tree_node({key: k,
                              mut value: v,
                              mut left: none,
                              mut right: none}));
        return;
      }
      some(node) {
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
fn find<K: copy, V: copy>(m: &const tree_edge<K, V>, k: K) -> option<V> {
    alt copy *m {
      none { none }

      // FIXME (#2808): was that an optimization?
      some(node) {
        if k == node.key {
            some(node.value)
        } else if k < node.key {
            find(&const node.left, k)
        } else {
            find(&const node.right, k)
        }
      }
    }
}

/// Visit all pairs in the map in order.
fn traverse<K, V: copy>(m: &const tree_edge<K, V>, f: fn(K, V)) {
    alt copy *m {
      none { }
      some(node) {
        traverse(&const node.left, f);
        // copy of value is req'd as f() requires an immutable ptr
        f(node.key, copy node.value);
        traverse(&const node.right, f);
      }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn init_treemap() { let _m = treemap::<int, int>(); }

    #[test]
    fn insert_one() { let m = treemap(); insert(m, 1, 2); }

    #[test]
    fn insert_two() { let m = treemap(); insert(m, 1, 2); insert(m, 3, 4); }

    #[test]
    fn insert_find() {
        let m = treemap();
        insert(m, 1, 2);
        assert (find(m, 1) == some(2));
    }

    #[test]
    fn find_empty() {
        let m = treemap::<int, int>(); assert (find(m, 1) == none);
    }

    #[test]
    fn find_not_found() {
        let m = treemap();
        insert(m, 1, 2);
        assert (find(m, 2) == none);
    }

    #[test]
    fn traverse_in_order() {
        let m = treemap();
        insert(m, 3, ());
        insert(m, 0, ());
        insert(m, 4, ());
        insert(m, 2, ());
        insert(m, 1, ());

        let n = @mut 0;
        fn t(n: @mut int, &&k: int, &&_v: ()) {
            assert (*n == k); *n += 1;
        }
        traverse(m, |x,y| t(n, x, y));
    }

    #[test]
    fn u8_map() {
        let m = treemap();

        let k1 = str::bytes(~"foo");
        let k2 = str::bytes(~"bar");

        insert(m, k1, ~"foo");
        insert(m, k2, ~"bar");

        assert (find(m, k2) == some(~"bar"));
        assert (find(m, k1) == some(~"foo"));
    }
}
