#![feature(plugin, inclusive_range_syntax)]
#![plugin(clippy)]

use std::collections::*;
use std::rc::Rc;

static STATIC: [usize; 4] = [ 0,  1,  8, 16 ];
const CONST: [usize; 4] = [ 0,  1,  8, 16 ];

#[warn(clippy)]
fn for_loop_over_option_and_result() {
    let option = Some(1);
    let result = option.ok_or("x not found");
    let v = vec![0,1,2];

    // check FOR_LOOP_OVER_OPTION lint
    for x in option {
        println!("{}", x);
    }

    // check FOR_LOOP_OVER_RESULT lint
    for x in result {
        println!("{}", x);
    }

    for x in option.ok_or("x not found") {
        println!("{}", x);
    }

    // make sure LOOP_OVER_NEXT lint takes precedence when next() is the last call in the chain
    for x in v.iter().next() {
        println!("{}", x);
    }

    // make sure we lint when next() is not the last call in the chain
    for x in v.iter().next().and(Some(0)) {
        println!("{}", x);
    }

    for x in v.iter().next().ok_or("x not found") {
        println!("{}", x);
    }

    // check for false positives

    // for loop false positive
    for x in v {
        println!("{}", x);
    }

    // while let false positive for Option
    while let Some(x) = option {
        println!("{}", x);
        break;
    }

    // while let false positive for Result
    while let Ok(x) = result {
        println!("{}", x);
        break;
    }
}

struct Unrelated(Vec<u8>);
impl Unrelated {
    fn next(&self) -> std::slice::Iter<u8> {
        self.0.iter()
    }

    fn iter(&self) -> std::slice::Iter<u8> {
        self.0.iter()
    }
}

#[warn(needless_range_loop, explicit_iter_loop, explicit_into_iter_loop, iter_next_loop, reverse_range_loop, explicit_counter_loop, for_kv_map)]
#[warn(unused_collect)]
#[allow(linkedlist, shadow_unrelated, unnecessary_mut_passed, cyclomatic_complexity, similar_names)]
#[allow(many_single_char_names)]
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

    for i in 0..vec.len() { let _ = vec[i]; }

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
    for i in 0..vec.len() {      // not an error, indexing more than one variable
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

    for i in 0...MAX_LEN {
        println!("{}", vec[i]);
    }

    for i in 5..10 {
        println!("{}", vec[i]);
    }

    for i in 5...10 {
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

    for i in 10...0 {
        println!("{}", i);
    }

    for i in MAX_LEN..0 {
        println!("{}", i);
    }

    for i in 5..5 {
        println!("{}", i);
    }

    for i in 5...5 { // not an error, this is the range with only one element “5”
        println!("{}", i);
    }

    for i in 0..10 { // not an error, the start index is less than the end index
        println!("{}", i);
    }

    for i in -10..0 { // not an error
        println!("{}", i);
    }

    for i in (10..0).map(|x| x * 2) { // not an error, it can't be known what arbitrary methods do to a range
        println!("{}", i);
    }

    // testing that the empty range lint folds constants
    for i in 10..5+4 {
        println!("{}", i);
    }

    for i in (5+2)..(3-1) {
        println!("{}", i);
    }

    for i in (5+2)..(8-1) {
        println!("{}", i);
    }

    for i in (2*2)..(2*3) { // no error, 4..6 is fine
        println!("{}", i);
    }

    let x = 42;
    for i in x..10 { // no error, not constant-foldable
        println!("{}", i);
    }

    // See #601
    for i in 0..10 { // no error, id_col does not exist outside the loop
        let mut id_col = vec![0f64; 10];
        id_col[i] = 1f64;
    }

    for _v in vec.iter() { }

    for _v in vec.iter_mut() { }

    let out_vec = vec![1,2,3];
    for _v in out_vec.into_iter() { }

    let array = [1, 2, 3];
    for _v in array.into_iter() {}

    for _v in &vec { } // these are fine
    for _v in &mut vec { } // these are fine

    for _v in [1, 2, 3].iter() { }

    for _v in (&mut [1, 2, 3]).iter() { } // no error

    for _v in [0; 32].iter() {}

    for _v in [0; 33].iter() {} // no error

    let ll: LinkedList<()> = LinkedList::new();
    for _v in ll.iter() { }

    let vd: VecDeque<()> = VecDeque::new();
    for _v in vd.iter() { }

    let bh: BinaryHeap<()> = BinaryHeap::new();
    for _v in bh.iter() { }

    let hm: HashMap<(), ()> = HashMap::new();
    for _v in hm.iter() { }

    let bt: BTreeMap<(), ()> = BTreeMap::new();
    for _v in bt.iter() { }

    let hs: HashSet<()> = HashSet::new();
    for _v in hs.iter() { }

    let bs: BTreeSet<()> = BTreeSet::new();
    for _v in bs.iter() { }

    for _v in vec.iter().next() { }

    let u = Unrelated(vec![]);
    for _v in u.next() { } // no error
    for _v in u.iter() { } // no error

    let mut out = vec![];
    vec.iter().cloned().map(|x| out.push(x)).collect::<Vec<_>>();
    let _y = vec.iter().cloned().map(|x| out.push(x)).collect::<Vec<_>>(); // this is fine

    // Loop with explicit counter variable
    let mut _index = 0;
    for _v in &vec { _index += 1 }

    let mut _index = 1;
    _index = 0;
    for _v in &vec { _index += 1 }

    // Potential false positives
    let mut _index = 0;
    _index = 1;
    for _v in &vec { _index += 1 }

    let mut _index = 0;
    _index += 1;
    for _v in &vec { _index += 1 }

    let mut _index = 0;
    if true { _index = 1 }
    for _v in &vec { _index += 1 }

    let mut _index = 0;
    let mut _index = 1;
    for _v in &vec { _index += 1 }

    let mut _index = 0;
    for _v in &vec { _index += 1; _index += 1 }

    let mut _index = 0;
    for _v in &vec { _index *= 2; _index += 1 }

    let mut _index = 0;
    for _v in &vec { _index = 1; _index += 1 }

    let mut _index = 0;

    for _v in &vec { let mut _index = 0; _index += 1 }

    let mut _index = 0;
    for _v in &vec { _index += 1; _index = 0; }

    let mut _index = 0;
    for _v in &vec { for _x in 0..1 { _index += 1; }; _index += 1 }

    let mut _index = 0;
    for x in &vec { if *x == 1 { _index += 1 } }

    let mut _index = 0;
    if true { _index = 1 };
    for _v in &vec { _index += 1 }

    let mut _index = 1;
    if false { _index = 0 };
    for _v in &vec { _index += 1 }

    let mut index = 0;
    { let mut _x = &mut index; }
    for _v in &vec { _index += 1 }

    let mut index = 0;
    for _v in &vec { index += 1 }
    println!("index: {}", index);

    for_loop_over_option_and_result();

    let m : HashMap<u64, u64> = HashMap::new();
    for (_, v) in &m {
        let _v = v;
    }

    let m : Rc<HashMap<u64, u64>> = Rc::new(HashMap::new());
    for (_, v) in &*m {
        let _v = v;
        // Here the `*` is not actually necesarry, but the test tests that we don't suggest
        // `in *m.values()` as we used to
    }

    let mut m : HashMap<u64, u64> = HashMap::new();
    for (_, v) in &mut m {
        let _v = v;
    }

    let m: &mut HashMap<u64, u64> = &mut HashMap::new();
    for (_, v) in &mut *m {
        let _v = v;
    }

    let m : HashMap<u64, u64> = HashMap::new();
    let rm = &m;
    for (k, _value) in rm {
        let _k = k;
    }

    test_for_kv_map();

    fn f<T>(_: &T, _: &T) -> bool { unimplemented!() }
    fn g<T>(_: &mut [T], _: usize, _: usize) { unimplemented!() }
    for i in 1..vec.len() {
        if f(&vec[i - 1], &vec[i]) {
            g(&mut vec, i - 1, i);
        }
    }

    for mid in 1..vec.len() {
        let (_, _) = vec.split_at(mid);
    }
}

#[allow(used_underscore_binding)]
fn test_for_kv_map() {
    let m : HashMap<u64, u64> = HashMap::new();

    // No error, _value is actually used
    for (k, _value) in &m {
        let _ = _value;
        let _k = k;
    }
}

#[allow(dead_code)]
fn partition<T:PartialOrd+Send>(v: &mut [T]) -> usize {
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
