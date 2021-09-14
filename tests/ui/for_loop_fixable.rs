// run-rustfix

#![allow(dead_code, unused)]

use std::collections::*;

#[warn(clippy::all)]
struct Unrelated(Vec<u8>);
impl Unrelated {
    fn next(&self) -> std::slice::Iter<u8> {
        self.0.iter()
    }

    fn iter(&self) -> std::slice::Iter<u8> {
        self.0.iter()
    }
}

#[warn(
    clippy::needless_range_loop,
    clippy::explicit_iter_loop,
    clippy::explicit_into_iter_loop,
    clippy::iter_next_loop,
    clippy::for_kv_map
)]
#[allow(
    clippy::linkedlist,
    clippy::shadow_unrelated,
    clippy::unnecessary_mut_passed,
    clippy::similar_names
)]
#[allow(unused_variables)]
fn main() {
    let mut vec = vec![1, 2, 3, 4];

    // See #601
    for i in 0..10 {
        // no error, id_col does not exist outside the loop
        let mut id_col = vec![0f64; 10];
        id_col[i] = 1f64;
    }

    for _v in vec.iter() {}

    for _v in vec.iter_mut() {}

    let out_vec = vec![1, 2, 3];
    for _v in out_vec.into_iter() {}

    for _v in &vec {} // these are fine
    for _v in &mut vec {} // these are fine

    for _v in [1, 2, 3].iter() {}

    for _v in (&mut [1, 2, 3]).iter() {} // no error

    for _v in [0; 32].iter() {}

    for _v in [0; 33].iter() {} // no error

    let ll: LinkedList<()> = LinkedList::new();
    for _v in ll.iter() {}

    let vd: VecDeque<()> = VecDeque::new();
    for _v in vd.iter() {}

    let bh: BinaryHeap<()> = BinaryHeap::new();
    for _v in bh.iter() {}

    let hm: HashMap<(), ()> = HashMap::new();
    for _v in hm.iter() {}

    let bt: BTreeMap<(), ()> = BTreeMap::new();
    for _v in bt.iter() {}

    let hs: HashSet<()> = HashSet::new();
    for _v in hs.iter() {}

    let bs: BTreeSet<()> = BTreeSet::new();
    for _v in bs.iter() {}

    let u = Unrelated(vec![]);
    for _v in u.next() {} // no error
    for _v in u.iter() {} // no error

    let mut out = vec![];
    vec.iter().cloned().map(|x| out.push(x)).collect::<Vec<_>>();
    let _y = vec.iter().cloned().map(|x| out.push(x)).collect::<Vec<_>>(); // this is fine

    // Loop with explicit counter variable

    // Potential false positives
    let mut _index = 0;
    _index = 1;
    for _v in &vec {
        _index += 1
    }

    let mut _index = 0;
    _index += 1;
    for _v in &vec {
        _index += 1
    }

    let mut _index = 0;
    if true {
        _index = 1
    }
    for _v in &vec {
        _index += 1
    }

    let mut _index = 0;
    let mut _index = 1;
    for _v in &vec {
        _index += 1
    }

    let mut _index = 0;
    for _v in &vec {
        _index += 1;
        _index += 1
    }

    let mut _index = 0;
    for _v in &vec {
        _index *= 2;
        _index += 1
    }

    let mut _index = 0;
    for _v in &vec {
        _index = 1;
        _index += 1
    }

    let mut _index = 0;

    for _v in &vec {
        let mut _index = 0;
        _index += 1
    }

    let mut _index = 0;
    for _v in &vec {
        _index += 1;
        _index = 0;
    }

    let mut _index = 0;
    for _v in &vec {
        for _x in 0..1 {
            _index += 1;
        }
        _index += 1
    }

    let mut _index = 0;
    for x in &vec {
        if *x == 1 {
            _index += 1
        }
    }

    let mut _index = 0;
    if true {
        _index = 1
    };
    for _v in &vec {
        _index += 1
    }

    let mut _index = 1;
    if false {
        _index = 0
    };
    for _v in &vec {
        _index += 1
    }

    let mut index = 0;
    {
        let mut _x = &mut index;
    }
    for _v in &vec {
        _index += 1
    }

    let mut index = 0;
    for _v in &vec {
        index += 1
    }
    println!("index: {}", index);

    fn f<T>(_: &T, _: &T) -> bool {
        unimplemented!()
    }
    fn g<T>(_: &mut [T], _: usize, _: usize) {
        unimplemented!()
    }
    for i in 1..vec.len() {
        if f(&vec[i - 1], &vec[i]) {
            g(&mut vec, i - 1, i);
        }
    }

    for mid in 1..vec.len() {
        let (_, _) = vec.split_at(mid);
    }
}

fn partition<T: PartialOrd + Send>(v: &mut [T]) -> usize {
    let pivot = v.len() - 1;
    let mut i = 0;
    for j in 0..pivot {
        if v[j] <= v[pivot] {
            v.swap(i, j);
            i += 1;
        }
    }
    v.swap(i, pivot);
    i
}

#[warn(clippy::needless_range_loop)]
pub fn manual_copy_same_destination(dst: &mut [i32], d: usize, s: usize) {
    // Same source and destination - don't trigger lint
    for i in 0..dst.len() {
        dst[d + i] = dst[s + i];
    }
}

mod issue_2496 {
    pub trait Handle {
        fn new_for_index(index: usize) -> Self;
        fn index(&self) -> usize;
    }

    pub fn test<H: Handle>() -> H {
        for x in 0..5 {
            let next_handle = H::new_for_index(x);
            println!("{}", next_handle.index());
        }
        unimplemented!()
    }
}

// explicit_into_iter_loop bad suggestions
#[warn(clippy::explicit_into_iter_loop, clippy::explicit_iter_loop)]
mod issue_4958 {
    fn takes_iterator<T>(iterator: &T)
    where
        for<'a> &'a T: IntoIterator<Item = &'a String>,
    {
        for i in iterator.into_iter() {
            println!("{}", i);
        }
    }

    struct T;
    impl IntoIterator for &T {
        type Item = ();
        type IntoIter = std::vec::IntoIter<Self::Item>;
        fn into_iter(self) -> Self::IntoIter {
            vec![].into_iter()
        }
    }

    fn more_tests() {
        let t = T;
        let r = &t;
        let rr = &&t;

        // This case is handled by `explicit_iter_loop`. No idea why.
        for _ in t.into_iter() {}

        for _ in r.into_iter() {}

        // No suggestion for this.
        // We'd have to suggest `for _ in *rr {}` which is less clear.
        for _ in rr.into_iter() {}
    }
}

// explicit_into_iter_loop
#[warn(clippy::explicit_into_iter_loop)]
mod issue_6900 {
    struct S;
    impl S {
        #[allow(clippy::should_implement_trait)]
        pub fn into_iter<T>(self) -> I<T> {
            unimplemented!()
        }
    }

    struct I<T>(T);
    impl<T> Iterator for I<T> {
        type Item = T;
        fn next(&mut self) -> Option<Self::Item> {
            unimplemented!()
        }
    }

    fn f() {
        for _ in S.into_iter::<u32>() {
            unimplemented!()
        }
    }
}
