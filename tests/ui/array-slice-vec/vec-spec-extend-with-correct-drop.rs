//@ run-pass
#![feature(trivial_clone)]

use std::cell::Cell;
use std::clone::TrivialClone;

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

    let mut vec = vec![DropCounter(&c); 3];
    vec.clear();

    assert_eq!(c.get(), 3);
    vec = vec![DropCounter(&c); 6];
    vec.clear();

    assert_eq!(c.get(), 9);

    vec = vec![DropCounter(&c); 0];
    assert_eq!(c.get(), 10);
    vec.clear();
    assert_eq!(c.get(), 10);

    // nontrivial clone
    let clones = Cell::new(0);
    let drops = Cell::new(0);

    let mut vec = vec![CloneCounter::new(&clones, &drops); 3];
    assert_eq!(clones.get(), 2);
    assert_eq!(drops.get(), 0);
    vec.clear();
    assert_eq!(clones.get(), 2);
    assert_eq!(drops.get(), 3);

    vec = vec![CloneCounter::new(&clones, &drops); 6];
    assert_eq!(clones.get(), 7);
    assert_eq!(drops.get(), 3);
    vec.clear();
    assert_eq!(clones.get(), 7);
    assert_eq!(drops.get(), 9);

    vec = vec![CloneCounter::new(&clones, &drops); 0];
    assert_eq!(clones.get(), 7);
    assert_eq!(drops.get(), 10);
    vec.clear();
    assert_eq!(clones.get(), 7);
    assert_eq!(drops.get(), 10);
}
