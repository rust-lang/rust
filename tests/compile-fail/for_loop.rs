#![feature(plugin)]
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

#[deny(needless_range_loop, explicit_iter_loop, iter_next_loop)]
#[deny(unused_collect)]
#[allow(linkedlist)]
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

    for _v in vec.iter() { } //~ERROR it is more idiomatic to loop over `&vec`
    for _v in vec.iter_mut() { } //~ERROR it is more idiomatic to loop over `&mut vec`

    for _v in &vec { } // these are fine
    for _v in &mut vec { } // these are fine

    for _v in [1, 2, 3].iter() { } //~ERROR it is more idiomatic to loop over `&[
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
}
