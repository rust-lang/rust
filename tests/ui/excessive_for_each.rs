#![warn(clippy::excessive_for_each)]
#![allow(clippy::needless_return)]

use std::collections::*;

fn main() {
    // Should trigger this lint: Vec.
    let vec: Vec<i32> = Vec::new();
    let mut acc = 0;
    vec.iter().for_each(|v| {
        acc += v;
    });

    // Should trigger this lint: &Vec.
    let vec_ref = &vec;
    vec_ref.iter().for_each(|v| {
        acc += v;
    });

    // Should trigger this lint: VecDeque.
    let vec_deq: VecDeque<i32> = VecDeque::new();
    vec_deq.iter().for_each(|v| {
        acc += v;
    });

    // Should trigger this lint: LinkedList.
    let list: LinkedList<i32> = LinkedList::new();
    list.iter().for_each(|v| {
        acc += v;
    });

    // Should trigger this lint: HashMap.
    let mut hash_map: HashMap<i32, i32> = HashMap::new();
    hash_map.iter().for_each(|(k, v)| {
        acc += k + v;
    });
    hash_map.iter_mut().for_each(|(k, v)| {
        acc += *k + *v;
    });
    hash_map.keys().for_each(|k| {
        acc += k;
    });
    hash_map.values().for_each(|v| {
        acc += v;
    });

    // Should trigger this lint: HashSet.
    let hash_set: HashSet<i32> = HashSet::new();
    hash_set.iter().for_each(|v| {
        acc += v;
    });

    // Should trigger this lint: BTreeSet.
    let btree_set: BTreeSet<i32> = BTreeSet::new();
    btree_set.iter().for_each(|v| {
        acc += v;
    });

    // Should trigger this lint: BinaryHeap.
    let binary_heap: BinaryHeap<i32> = BinaryHeap::new();
    binary_heap.iter().for_each(|v| {
        acc += v;
    });

    // Should trigger this lint: Array.
    let s = [1, 2, 3];
    s.iter().for_each(|v| {
        acc += v;
    });

    // Should trigger this lint. Slice.
    vec.as_slice().iter().for_each(|v| {
        acc += v;
    });

    // Should trigger this lint with notes that say "change `return` to `continue`".
    vec.iter().for_each(|v| {
        if *v == 10 {
            return;
        } else {
            println!("{}", v);
        }
    });

    // Should trigger this lint with notes that say "change `return` to `continue 'outer`".
    vec.iter().for_each(|v| {
        for i in 0..*v {
            if i == 10 {
                return;
            } else {
                println!("{}", v);
            }
        }
        if *v == 20 {
            return;
        } else {
            println!("{}", v);
        }
    });

    // Should NOT trigger this lint in case `for_each` follows long iterator chain.
    vec.iter().chain(vec.iter()).for_each(|v| println!("{}", v));

    // Should NOT trigger this lint in case a `for_each` argument is not closure.
    fn print(x: &i32) {
        println!("{}", x);
    }
    vec.iter().for_each(print);

    // Should NOT trigger this lint in case the receiver of `iter` is a user defined type.
    let my_collection = MyCollection { v: vec![] };
    my_collection.iter().for_each(|v| println!("{}", v));

    // Should NOT trigger this lint in case the closure body is not a `ExprKind::Block`.
    vec.iter().for_each(|x| acc += x);
}

struct MyCollection {
    v: Vec<i32>,
}

impl MyCollection {
    fn iter(&self) -> impl Iterator<Item = &i32> {
        self.v.iter()
    }
}
