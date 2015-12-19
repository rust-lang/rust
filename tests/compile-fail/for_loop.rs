#![feature(plugin, step_by)]
#![plugin(clippy)]

use std::collections::*;

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
#[allow(linkedlist,shadow_unrelated,unnecessary_mut_passed, cyclomatic_complexity)]
fn main() {
    let mut vec = vec![1, 2, 3, 4];
    let vec2 = vec![1, 2, 3, 4];
    for i in 0..vec.len() {      //~ERROR the loop variable `i` is only used to index `vec`.
        println!("{}", vec[i]);
    }
    for i in 0..vec.len() {      //~ERROR the loop variable `i` is used to index `vec`.
        println!("{} {}", vec[i], i);
    }
    for i in 0..vec.len() {      // not an error, indexing more than one variable
        println!("{} {}", vec[i], vec2[i]);
    }

    for i in 5..vec.len() {      // not an error, not starting with 0
        println!("{}", vec[i]);
    }

    for i in 10..0 { //~ERROR this range is empty so this for loop will never run
        println!("{}", i);
    }

    for i in 5..5 { //~ERROR this range is empty so this for loop will never run
        println!("{}", i);
    }

    for i in 0..10 { // not an error, the start index is less than the end index
        println!("{}", i);
    }

    for i in -10..0 { // not an error
        println!("{}", i);
    }

    for i in (10..0).rev() { // not an error, this is an established idiom for looping backwards on a range
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
}
