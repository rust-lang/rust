//@run-rustfix
#![warn(clippy::manual_retain)]
#![allow(unused, clippy::redundant_clone)]
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};

fn main() {
    binary_heap_retain();
    btree_set_retain();
    btree_map_retain();
    hash_set_retain();
    hash_map_retain();
    string_retain();
    vec_deque_retain();
    vec_retain();
    _msrv_153();
    _msrv_126();
    _msrv_118();
}

fn binary_heap_retain() {
    // NOTE: Do not lint now, because binary_heap_retain is nightly API.
    // And we need to add a test case for msrv if we update this implementation.
    // https://github.com/rust-lang/rust/issues/71503
    let mut heap = BinaryHeap::from([1, 2, 3]);
    heap = heap.into_iter().filter(|x| x % 2 == 0).collect();
    heap = heap.iter().filter(|&x| x % 2 == 0).copied().collect();
    heap = heap.iter().filter(|&x| x % 2 == 0).cloned().collect();

    // Do not lint, because type conversion is performed
    heap = heap.into_iter().filter(|x| x % 2 == 0).collect::<BinaryHeap<i8>>();
    heap = heap.iter().filter(|&x| x % 2 == 0).copied().collect::<BinaryHeap<i8>>();
    heap = heap.iter().filter(|&x| x % 2 == 0).cloned().collect::<BinaryHeap<i8>>();

    // Do not lint, because this expression is not assign.
    let mut bar: BinaryHeap<i8> = heap.iter().filter(|&x| x % 2 == 0).copied().collect();
    let mut foobar: BinaryHeap<i8> = heap.into_iter().filter(|x| x % 2 == 0).collect();

    // Do not lint, because it is an assignment to a different variable.
    bar = foobar.iter().filter(|&x| x % 2 == 0).copied().collect();
    bar = foobar.into_iter().filter(|x| x % 2 == 0).collect();
}

fn btree_map_retain() {
    let mut btree_map: BTreeMap<i8, i8> = (0..8).map(|x| (x, x * 10)).collect();
    // Do lint.
    btree_map = btree_map.into_iter().filter(|(k, _)| k % 2 == 0).collect();
    btree_map = btree_map.into_iter().filter(|(_, v)| v % 2 == 0).collect();
    btree_map = btree_map
        .into_iter()
        .filter(|(k, v)| (k % 2 == 0) && (v % 2 == 0))
        .collect();

    // Do not lint.
    btree_map = btree_map
        .into_iter()
        .filter(|(x, _)| x % 2 == 0)
        .collect::<BTreeMap<i8, i8>>();

    // Do not lint, because this expression is not assign.
    let mut foobar: BTreeMap<i8, i8> = btree_map.into_iter().filter(|(k, _)| k % 2 == 0).collect();

    // Do not lint, because it is an assignment to a different variable.
    btree_map = foobar.into_iter().filter(|(k, _)| k % 2 == 0).collect();
}

fn btree_set_retain() {
    let mut btree_set = BTreeSet::from([1, 2, 3, 4, 5, 6]);

    // Do lint.
    btree_set = btree_set.iter().filter(|&x| x % 2 == 0).copied().collect();
    btree_set = btree_set.iter().filter(|&x| x % 2 == 0).cloned().collect();
    btree_set = btree_set.into_iter().filter(|x| x % 2 == 0).collect();

    // Do not lint, because type conversion is performed
    btree_set = btree_set
        .iter()
        .filter(|&x| x % 2 == 0)
        .copied()
        .collect::<BTreeSet<i8>>();

    btree_set = btree_set
        .iter()
        .filter(|&x| x % 2 == 0)
        .cloned()
        .collect::<BTreeSet<i8>>();

    btree_set = btree_set.into_iter().filter(|x| x % 2 == 0).collect::<BTreeSet<i8>>();

    // Do not lint, because this expression is not assign.
    let mut foobar: BTreeSet<i8> = btree_set.iter().filter(|&x| x % 2 == 0).copied().collect();
    let mut bar: BTreeSet<i8> = btree_set.into_iter().filter(|x| x % 2 == 0).collect();

    // Do not lint, because it is an assignment to a different variable.
    bar = foobar.iter().filter(|&x| x % 2 == 0).copied().collect();
    bar = foobar.iter().filter(|&x| x % 2 == 0).cloned().collect();
    bar = foobar.into_iter().filter(|x| x % 2 == 0).collect();
}

fn hash_map_retain() {
    let mut hash_map: HashMap<i8, i8> = (0..8).map(|x| (x, x * 10)).collect();
    // Do lint.
    hash_map = hash_map.into_iter().filter(|(k, _)| k % 2 == 0).collect();
    hash_map = hash_map.into_iter().filter(|(_, v)| v % 2 == 0).collect();
    hash_map = hash_map
        .into_iter()
        .filter(|(k, v)| (k % 2 == 0) && (v % 2 == 0))
        .collect();

    // Do not lint.
    hash_map = hash_map
        .into_iter()
        .filter(|(x, _)| x % 2 == 0)
        .collect::<HashMap<i8, i8>>();

    // Do not lint, because this expression is not assign.
    let mut foobar: HashMap<i8, i8> = hash_map.into_iter().filter(|(k, _)| k % 2 == 0).collect();

    // Do not lint, because it is an assignment to a different variable.
    hash_map = foobar.into_iter().filter(|(k, _)| k % 2 == 0).collect();
}

fn hash_set_retain() {
    let mut hash_set = HashSet::from([1, 2, 3, 4, 5, 6]);
    // Do lint.
    hash_set = hash_set.into_iter().filter(|x| x % 2 == 0).collect();
    hash_set = hash_set.iter().filter(|&x| x % 2 == 0).copied().collect();
    hash_set = hash_set.iter().filter(|&x| x % 2 == 0).cloned().collect();

    // Do not lint, because type conversion is performed
    hash_set = hash_set.into_iter().filter(|x| x % 2 == 0).collect::<HashSet<i8>>();
    hash_set = hash_set
        .iter()
        .filter(|&x| x % 2 == 0)
        .copied()
        .collect::<HashSet<i8>>();

    hash_set = hash_set
        .iter()
        .filter(|&x| x % 2 == 0)
        .cloned()
        .collect::<HashSet<i8>>();

    // Do not lint, because this expression is not assign.
    let mut bar: HashSet<i8> = hash_set.iter().filter(|&x| x % 2 == 0).copied().collect();
    let mut foobar: HashSet<i8> = hash_set.into_iter().filter(|x| x % 2 == 0).collect();

    // Do not lint, because it is an assignment to a different variable.
    bar = foobar.iter().filter(|&x| x % 2 == 0).copied().collect();
    bar = foobar.iter().filter(|&x| x % 2 == 0).cloned().collect();
    bar = foobar.into_iter().filter(|&x| x % 2 == 0).collect();
}

fn string_retain() {
    let mut s = String::from("foobar");
    // Do lint.
    s = s.chars().filter(|&c| c != 'o').to_owned().collect();

    // Do not lint, because this expression is not assign.
    let mut bar: String = s.chars().filter(|&c| c != 'o').to_owned().collect();

    // Do not lint, because it is an assignment to a different variable.
    s = bar.chars().filter(|&c| c != 'o').to_owned().collect();
}

fn vec_retain() {
    let mut vec = vec![0, 1, 2];
    // Do lint.
    vec = vec.iter().filter(|&x| x % 2 == 0).copied().collect();
    vec = vec.iter().filter(|&x| x % 2 == 0).cloned().collect();
    vec = vec.into_iter().filter(|x| x % 2 == 0).collect();

    // Do not lint, because type conversion is performed
    vec = vec.into_iter().filter(|x| x % 2 == 0).collect::<Vec<i8>>();
    vec = vec.iter().filter(|&x| x % 2 == 0).copied().collect::<Vec<i8>>();
    vec = vec.iter().filter(|&x| x % 2 == 0).cloned().collect::<Vec<i8>>();

    // Do not lint, because this expression is not assign.
    let mut bar: Vec<i8> = vec.iter().filter(|&x| x % 2 == 0).copied().collect();
    let mut foobar: Vec<i8> = vec.into_iter().filter(|x| x % 2 == 0).collect();

    // Do not lint, because it is an assignment to a different variable.
    bar = foobar.iter().filter(|&x| x % 2 == 0).copied().collect();
    bar = foobar.iter().filter(|&x| x % 2 == 0).cloned().collect();
    bar = foobar.into_iter().filter(|x| x % 2 == 0).collect();
}

fn vec_deque_retain() {
    let mut vec_deque = VecDeque::new();
    vec_deque.extend(1..5);

    // Do lint.
    vec_deque = vec_deque.iter().filter(|&x| x % 2 == 0).copied().collect();
    vec_deque = vec_deque.iter().filter(|&x| x % 2 == 0).cloned().collect();
    vec_deque = vec_deque.into_iter().filter(|x| x % 2 == 0).collect();

    // Do not lint, because type conversion is performed
    vec_deque = vec_deque
        .iter()
        .filter(|&x| x % 2 == 0)
        .copied()
        .collect::<VecDeque<i8>>();
    vec_deque = vec_deque
        .iter()
        .filter(|&x| x % 2 == 0)
        .cloned()
        .collect::<VecDeque<i8>>();
    vec_deque = vec_deque.into_iter().filter(|x| x % 2 == 0).collect::<VecDeque<i8>>();

    // Do not lint, because this expression is not assign.
    let mut bar: VecDeque<i8> = vec_deque.iter().filter(|&x| x % 2 == 0).copied().collect();
    let mut foobar: VecDeque<i8> = vec_deque.into_iter().filter(|x| x % 2 == 0).collect();

    // Do not lint, because it is an assignment to a different variable.
    bar = foobar.iter().filter(|&x| x % 2 == 0).copied().collect();
    bar = foobar.iter().filter(|&x| x % 2 == 0).cloned().collect();
    bar = foobar.into_iter().filter(|x| x % 2 == 0).collect();
}

#[clippy::msrv = "1.52"]
fn _msrv_153() {
    let mut btree_map: BTreeMap<i8, i8> = (0..8).map(|x| (x, x * 10)).collect();
    btree_map = btree_map.into_iter().filter(|(k, _)| k % 2 == 0).collect();

    let mut btree_set = BTreeSet::from([1, 2, 3, 4, 5, 6]);
    btree_set = btree_set.iter().filter(|&x| x % 2 == 0).copied().collect();
}

#[clippy::msrv = "1.25"]
fn _msrv_126() {
    let mut s = String::from("foobar");
    s = s.chars().filter(|&c| c != 'o').to_owned().collect();
}

#[clippy::msrv = "1.17"]
fn _msrv_118() {
    let mut hash_set = HashSet::from([1, 2, 3, 4, 5, 6]);
    hash_set = hash_set.into_iter().filter(|x| x % 2 == 0).collect();
    let mut hash_map: HashMap<i8, i8> = (0..8).map(|x| (x, x * 10)).collect();
    hash_map = hash_map.into_iter().filter(|(k, _)| k % 2 == 0).collect();
}
