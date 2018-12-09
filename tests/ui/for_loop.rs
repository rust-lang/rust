// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::*;
use std::rc::Rc;

static STATIC: [usize; 4] = [0, 1, 8, 16];
const CONST: [usize; 4] = [0, 1, 8, 16];

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
    clippy::reverse_range_loop,
    clippy::for_kv_map
)]
#[warn(clippy::unused_collect)]
#[allow(
    clippy::linkedlist,
    clippy::shadow_unrelated,
    clippy::unnecessary_mut_passed,
    clippy::cyclomatic_complexity,
    clippy::similar_names
)]
#[allow(clippy::many_single_char_names, unused_variables, clippy::into_iter_on_array)]
fn main() {
    const MAX_LEN: usize = 42;

    let mut vec = vec![1, 2, 3, 4];
    let vec2 = vec![1, 2, 3, 4];
    for i in 0..vec.len() {
        println!("{}", vec[i]);
    }

    for i in 0..vec.len() {
        let i = 42; // make a different `i`
        println!("{}", vec[i]); // ok, not the `i` of the for-loop
    }

    for i in 0..vec.len() {
        let _ = vec[i];
    }

    // ICE #746
    for j in 0..4 {
        println!("{:?}", STATIC[j]);
    }

    for j in 0..4 {
        println!("{:?}", CONST[j]);
    }

    for i in 0..vec.len() {
        println!("{} {}", vec[i], i);
    }
    for i in 0..vec.len() {
        // not an error, indexing more than one variable
        println!("{} {}", vec[i], vec2[i]);
    }

    for i in 0..vec.len() {
        println!("{}", vec2[i]);
    }

    for i in 5..vec.len() {
        println!("{}", vec[i]);
    }

    for i in 0..MAX_LEN {
        println!("{}", vec[i]);
    }

    for i in 0..=MAX_LEN {
        println!("{}", vec[i]);
    }

    for i in 5..10 {
        println!("{}", vec[i]);
    }

    for i in 5..=10 {
        println!("{}", vec[i]);
    }

    for i in 5..vec.len() {
        println!("{} {}", vec[i], i);
    }

    for i in 5..10 {
        println!("{} {}", vec[i], i);
    }

    for i in 10..0 {
        println!("{}", i);
    }

    for i in 10..=0 {
        println!("{}", i);
    }

    for i in MAX_LEN..0 {
        println!("{}", i);
    }

    for i in 5..5 {
        println!("{}", i);
    }

    for i in 5..=5 {
        // not an error, this is the range with only one element “5”
        println!("{}", i);
    }

    for i in 0..10 {
        // not an error, the start index is less than the end index
        println!("{}", i);
    }

    for i in -10..0 {
        // not an error
        println!("{}", i);
    }

    for i in (10..0).map(|x| x * 2) {
        // not an error, it can't be known what arbitrary methods do to a range
        println!("{}", i);
    }

    // testing that the empty range lint folds constants
    for i in 10..5 + 4 {
        println!("{}", i);
    }

    for i in (5 + 2)..(3 - 1) {
        println!("{}", i);
    }

    for i in (5 + 2)..(8 - 1) {
        println!("{}", i);
    }

    for i in (2 * 2)..(2 * 3) {
        // no error, 4..6 is fine
        println!("{}", i);
    }

    let x = 42;
    for i in x..10 {
        // no error, not constant-foldable
        println!("{}", i);
    }

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

    let array = [1, 2, 3];
    for _v in array.into_iter() {}

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

    for _v in vec.iter().next() {}

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

    let m: HashMap<u64, u64> = HashMap::new();
    for (_, v) in &m {
        let _v = v;
    }

    let m: Rc<HashMap<u64, u64>> = Rc::new(HashMap::new());
    for (_, v) in &*m {
        let _v = v;
        // Here the `*` is not actually necessary, but the test tests that we don't
        // suggest
        // `in *m.values()` as we used to
    }

    let mut m: HashMap<u64, u64> = HashMap::new();
    for (_, v) in &mut m {
        let _v = v;
    }

    let m: &mut HashMap<u64, u64> = &mut HashMap::new();
    for (_, v) in &mut *m {
        let _v = v;
    }

    let m: HashMap<u64, u64> = HashMap::new();
    let rm = &m;
    for (k, _value) in rm {
        let _k = k;
    }

    test_for_kv_map();

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

#[allow(clippy::used_underscore_binding)]
fn test_for_kv_map() {
    let m: HashMap<u64, u64> = HashMap::new();

    // No error, _value is actually used
    for (k, _value) in &m {
        let _ = _value;
        let _k = k;
    }
}

#[allow(dead_code)]
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

const LOOP_OFFSET: usize = 5000;

#[warn(clippy::needless_range_loop)]
pub fn manual_copy(src: &[i32], dst: &mut [i32], dst2: &mut [i32]) {
    // plain manual memcpy
    for i in 0..src.len() {
        dst[i] = src[i];
    }

    // dst offset memcpy
    for i in 0..src.len() {
        dst[i + 10] = src[i];
    }

    // src offset memcpy
    for i in 0..src.len() {
        dst[i] = src[i + 10];
    }

    // src offset memcpy
    for i in 11..src.len() {
        dst[i] = src[i - 10];
    }

    // overwrite entire dst
    for i in 0..dst.len() {
        dst[i] = src[i];
    }

    // manual copy with branch - can't easily convert to memcpy!
    for i in 0..src.len() {
        dst[i] = src[i];
        if dst[i] > 5 {
            break;
        }
    }

    // multiple copies - suggest two memcpy statements
    for i in 10..256 {
        dst[i] = src[i - 5];
        dst2[i + 500] = src[i]
    }

    // this is a reversal - the copy lint shouldn't be triggered
    for i in 10..LOOP_OFFSET {
        dst[i + LOOP_OFFSET] = src[LOOP_OFFSET - i];
    }

    let some_var = 5;
    // Offset in variable
    for i in 10..LOOP_OFFSET {
        dst[i + LOOP_OFFSET] = src[i - some_var];
    }

    // Non continuous copy - don't trigger lint
    for i in 0..10 {
        dst[i + i] = src[i];
    }

    let src_vec = vec![1, 2, 3, 4, 5];
    let mut dst_vec = vec![0, 0, 0, 0, 0];

    // make sure vectors are supported
    for i in 0..src_vec.len() {
        dst_vec[i] = src_vec[i];
    }

    // lint should not trigger when either
    // source or destination type is not
    // slice-like, like DummyStruct
    struct DummyStruct(i32);

    impl ::std::ops::Index<usize> for DummyStruct {
        type Output = i32;

        fn index(&self, _: usize) -> &i32 {
            &self.0
        }
    }

    let src = DummyStruct(5);
    let mut dst_vec = vec![0; 10];

    for i in 0..10 {
        dst_vec[i] = src[i];
    }

    // Simplify suggestion (issue #3004)
    let src = [0, 1, 2, 3, 4];
    let mut dst = [0, 0, 0, 0, 0, 0];
    let from = 1;

    for i in from..from + src.len() {
        dst[i] = src[i - from];
    }

    for i in from..from + 3 {
        dst[i] = src[i - from];
    }
}

#[warn(clippy::needless_range_loop)]
pub fn manual_clone(src: &[String], dst: &mut [String]) {
    for i in 0..src.len() {
        dst[i] = src[i].clone();
    }
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
