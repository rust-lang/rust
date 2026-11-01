//@ run-pass
#![feature(trivial_clone)]

use std::cell::Cell;
use std::clone::TrivialClone;
use std::vec::from_elem;

#[derive(Clone)]
struct DropCounter<'a>(&'a Cell<i32>);

unsafe impl<'a> TrivialClone for DropCounter<'a> {}

impl Drop for DropCounter<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

struct CloneCounter<'a> {
    clones: &'a Cell<i32>,
    drops: &'a Cell<i32>,
}

impl<'a> CloneCounter<'a> {
    fn new(clones: &'a Cell<i32>, drops: &'a Cell<i32>) -> Self {
        Self { clones, drops }
    }
}

impl<'a> Clone for CloneCounter<'a> {
    fn clone(&self) -> Self {
        self.clones.set(self.clones.get() + 1);
        Self::new(self.clones, self.drops)
    }
}

impl<'a> Drop for CloneCounter<'a> {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

fn main() {
    // TrivialClone
    let c = Cell::new(0);

    let mut vec = from_elem(DropCounter(&c), 3);
    vec.clear();

    assert_eq!(c.get(), 3);
    vec = from_elem(DropCounter(&c), 6);
    vec.clear();

    assert_eq!(c.get(), 9);

    vec = from_elem(DropCounter(&c), 0);
    assert_eq!(c.get(), 10);
    vec.clear();
    assert_eq!(c.get(), 10);

    // nontrivial clone
    let clones = Cell::new(0);
    let drops = Cell::new(0);

    let mut vec = from_elem(CloneCounter::new(&clones, &drops), 3);
    assert_eq!(clones.get(), 2);
    assert_eq!(drops.get(), 0);
    vec.clear();
    assert_eq!(clones.get(), 2);
    assert_eq!(drops.get(), 3);

    vec = from_elem(CloneCounter::new(&clones, &drops), 6);
    assert_eq!(clones.get(), 7);
    assert_eq!(drops.get(), 3);
    vec.clear();
    assert_eq!(clones.get(), 7);
    assert_eq!(drops.get(), 9);

    vec = from_elem(CloneCounter::new(&clones, &drops), 0);
    assert_eq!(clones.get(), 7);
    assert_eq!(drops.get(), 10);
    vec.clear();
    assert_eq!(clones.get(), 7);
    assert_eq!(drops.get(), 10);
}
