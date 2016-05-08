#![feature(plugin, step_by, inclusive_range_syntax)]
#![plugin(clippy)]

use std::collections::*;

static STATIC: [usize; 4] = [ 0,  1,  8, 16 ];
const CONST: [usize; 4] = [ 0,  1,  8, 16 ];

#[deny(clippy)]
fn for_loop_over_option_and_result() {
    let option = Some(1);
    let result = option.ok_or("x not found");
    let v = vec![0,1,2];

    // check FOR_LOOP_OVER_OPTION lint

    for x in option {
        //~^ ERROR for loop over `option`, which is an `Option`.
        //~| HELP consider replacing `for x in option` with `if let Some(x) = option`
        println!("{}", x);
    }

    // check FOR_LOOP_OVER_RESULT lint

    for x in result {
        //~^ ERROR for loop over `result`, which is a `Result`.
        //~| HELP consider replacing `for x in result` with `if let Ok(x) = result`
        println!("{}", x);
    }

    for x in option.ok_or("x not found") {
        //~^ ERROR for loop over `option.ok_or("x not found")`, which is a `Result`.
        //~| HELP consider replacing `for x in option.ok_or("x not found")` with `if let Ok(x) = option.ok_or("x not found")`
        println!("{}", x);
    }

    // make sure LOOP_OVER_NEXT lint takes precedence when next() is the last call in the chain

    for x in v.iter().next() {
        //~^ ERROR you are iterating over `Iterator::next()` which is an Option
        println!("{}", x);
    }

    // make sure we lint when next() is not the last call in the chain

    for x in v.iter().next().and(Some(0)) {
        //~^ ERROR for loop over `v.iter().next().and(Some(0))`, which is an `Option`
        //~| HELP consider replacing `for x in v.iter().next().and(Some(0))` with `if let Some(x) = v.iter().next().and(Some(0))`
        println!("{}", x);
    }

    for x in v.iter().next().ok_or("x not found") {
        //~^ ERROR for loop over `v.iter().next().ok_or("x not found")`, which is a `Result`
        //~| HELP consider replacing `for x in v.iter().next().ok_or("x not found")` with `if let Ok(x) = v.iter().next().ok_or("x not found")`
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

#[deny(needless_range_loop, explicit_iter_loop, iter_next_loop, reverse_range_loop, explicit_counter_loop)]
#[deny(unused_collect)]
#[allow(linkedlist, shadow_unrelated, unnecessary_mut_passed, cyclomatic_complexity, similar_names)]
#[allow(many_single_char_names)]
fn main() {
    const MAX_LEN: usize = 42;

    let mut vec = vec![1, 2, 3, 4];
    let vec2 = vec![1, 2, 3, 4];
    for i in 0..vec.len() {
        //~^ ERROR `i` is only used to index `vec`. Consider using `for item in &vec`
        println!("{}", vec[i]);
    }

    // ICE #746
    for j in 0..4 {
        //~^ ERROR `j` is only used to index `STATIC`
        println!("{:?}", STATIC[j]);
    }

    for j in 0..4 {
        //~^ ERROR `j` is only used to index `CONST`
        println!("{:?}", CONST[j]);
    }

    for i in 0..vec.len() {
        //~^ ERROR `i` is used to index `vec`. Consider using `for (i, item) in vec.iter().enumerate()`
        println!("{} {}", vec[i], i);
    }
    for i in 0..vec.len() {      // not an error, indexing more than one variable
        println!("{} {}", vec[i], vec2[i]);
    }

    for i in 0..vec.len() {
        //~^ ERROR `i` is only used to index `vec2`. Consider using `for item in vec2.iter().take(vec.len())`
        println!("{}", vec2[i]);
    }

    for i in 5..vec.len() {
        //~^ ERROR `i` is only used to index `vec`. Consider using `for item in vec.iter().skip(5)`
        println!("{}", vec[i]);
    }

    for i in 0..MAX_LEN {
        //~^ ERROR `i` is only used to index `vec`. Consider using `for item in vec.iter().take(MAX_LEN)`
        println!("{}", vec[i]);
    }

    for i in 0...MAX_LEN {
        //~^ ERROR `i` is only used to index `vec`. Consider using `for item in vec.iter().take(MAX_LEN)`
        println!("{}", vec[i]);
    }

    for i in 5..10 {
        //~^ ERROR `i` is only used to index `vec`. Consider using `for item in vec.iter().take(10).skip(5)`
        println!("{}", vec[i]);
    }

    for i in 5...10 {
        //~^ ERROR `i` is only used to index `vec`. Consider using `for item in vec.iter().take(10).skip(5)`
        println!("{}", vec[i]);
    }

    for i in 5..vec.len() {
        //~^ ERROR `i` is used to index `vec`. Consider using `for (i, item) in vec.iter().enumerate().skip(5)`
        println!("{} {}", vec[i], i);
    }

    for i in 5..10 {
        //~^ ERROR `i` is used to index `vec`. Consider using `for (i, item) in vec.iter().enumerate().take(10).skip(5)`
        println!("{} {}", vec[i], i);
    }

    for i in 10..0 {
        //~^ERROR this range is empty so this for loop will never run
        //~|HELP consider
        //~|SUGGESTION (0..10).rev()
        println!("{}", i);
    }

    for i in 10...0 {
        //~^ERROR this range is empty so this for loop will never run
        //~|HELP consider
        //~|SUGGESTION (0..10).rev()
        println!("{}", i);
    }

    for i in MAX_LEN..0 { //~ERROR this range is empty so this for loop will never run
        //~|HELP consider
        //~|SUGGESTION (0..MAX_LEN).rev()
        println!("{}", i);
    }

    for i in 5..5 { //~ERROR this range is empty so this for loop will never run
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
    for i in 10..5+4 { //~ERROR this range is empty so this for loop will never run
        println!("{}", i);
    }

    for i in (5+2)..(3-1) { //~ERROR this range is empty so this for loop will never run
        println!("{}", i);
    }

    for i in (5+2)..(8-1) { //~ERROR this range is empty so this for loop will never run
        println!("{}", i);
    }

    for i in (2*2)..(2*3) { // no error, 4..6 is fine
        println!("{}", i);
    }

    for i in (10..8).step_by(-1) {
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

    /*
    for i in (10..0).map(|x| x * 2) {
        println!("{}", i);
    }*/

    for _v in vec.iter() { } //~ERROR it is more idiomatic to loop over `&vec`
    for _v in vec.iter_mut() { } //~ERROR it is more idiomatic to loop over `&mut vec`

    for _v in &vec { } // these are fine
    for _v in &mut vec { } // these are fine

    for _v in [1, 2, 3].iter() { } //~ERROR it is more idiomatic to loop over `&[
    for _v in (&mut [1, 2, 3]).iter() { } // no error
    for _v in [0; 32].iter() {} //~ERROR it is more idiomatic to loop over `&[
    for _v in [0; 33].iter() {} // no error
    let ll: LinkedList<()> = LinkedList::new();
    for _v in ll.iter() { } //~ERROR it is more idiomatic to loop over `&ll`
    let vd: VecDeque<()> = VecDeque::new();
    for _v in vd.iter() { } //~ERROR it is more idiomatic to loop over `&vd`
    let bh: BinaryHeap<()> = BinaryHeap::new();
    for _v in bh.iter() { } //~ERROR it is more idiomatic to loop over `&bh`
    let hm: HashMap<(), ()> = HashMap::new();
    for _v in hm.iter() { } //~ERROR it is more idiomatic to loop over `&hm`
    let bt: BTreeMap<(), ()> = BTreeMap::new();
    for _v in bt.iter() { } //~ERROR it is more idiomatic to loop over `&bt`
    let hs: HashSet<()> = HashSet::new();
    for _v in hs.iter() { } //~ERROR it is more idiomatic to loop over `&hs`
    let bs: BTreeSet<()> = BTreeSet::new();
    for _v in bs.iter() { } //~ERROR it is more idiomatic to loop over `&bs`

    for _v in vec.iter().next() { } //~ERROR you are iterating over `Iterator::next()`

    let u = Unrelated(vec![]);
    for _v in u.next() { } // no error
    for _v in u.iter() { } // no error

    let mut out = vec![];
    vec.iter().map(|x| out.push(x)).collect::<Vec<_>>(); //~ERROR you are collect()ing an iterator
    let _y = vec.iter().map(|x| out.push(x)).collect::<Vec<_>>(); // this is fine

    // Loop with explicit counter variable
    let mut _index = 0;
    for _v in &vec { _index += 1 } //~ERROR the variable `_index` is used as a loop counter

    let mut _index = 1;
    _index = 0;
    for _v in &vec { _index += 1 } //~ERROR the variable `_index` is used as a loop counter

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
        //~^ you seem to want to iterate on a map's values
        //~| HELP use the corresponding method
        //~| SUGGESTION for v in m.values()
        let _v = v;
    }

    let mut m : HashMap<u64, u64> = HashMap::new();
    for (_, v) in &mut m {
        // Ok, there is no values_mut method or equivalent
        let _v = v;
    }


    let rm = &m;
    for (k, _value) in rm {
        //~^ you seem to want to iterate on a map's keys
        //~| HELP use the corresponding method
        //~| SUGGESTION for k in rm.keys()
        let _k = k;
    }

    test_for_kv_map();
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
