// run-pass
#![allow(unused_must_use)]
// Tests that a heterogeneous list of existential `dyn` types can be put inside an Arc
// and shared between threads as long as all types fulfill Send.

// ignore-emscripten no threads support

use std::sync::Arc;
use std::sync::mpsc::channel;
use std::thread;

trait Pet {
    fn name(&self, blk: Box<dyn FnMut(&str)>);
    fn num_legs(&self) -> usize;
    fn of_good_pedigree(&self) -> bool;
}

struct Catte {
    num_whiskers: usize,
    name: String,
}

struct Dogge {
    bark_decibels: usize,
    tricks_known: usize,
    name: String,
}

struct Goldfyshe {
    swim_speed: usize,
    name: String,
}

impl Pet for Catte {
    fn name(&self, mut blk: Box<dyn FnMut(&str)>) { blk(&self.name) }
    fn num_legs(&self) -> usize { 4 }
    fn of_good_pedigree(&self) -> bool { self.num_whiskers >= 4 }
}
impl Pet for Dogge {
    fn name(&self, mut blk: Box<dyn FnMut(&str)>) { blk(&self.name) }
    fn num_legs(&self) -> usize { 4 }
    fn of_good_pedigree(&self) -> bool {
        self.bark_decibels < 70 || self.tricks_known > 20
    }
}
impl Pet for Goldfyshe {
    fn name(&self, mut blk: Box<dyn FnMut(&str)>) { blk(&self.name) }
    fn num_legs(&self) -> usize { 0 }
    fn of_good_pedigree(&self) -> bool { self.swim_speed >= 500 }
}

pub fn main() {
    let catte = Catte { num_whiskers: 7, name: "alonzo_church".to_string() };
    let dogge1 = Dogge {
        bark_decibels: 100,
        tricks_known: 42,
        name: "alan_turing".to_string(),
    };
    let dogge2 = Dogge {
        bark_decibels: 55,
        tricks_known: 11,
        name: "albert_einstein".to_string(),
    };
    let fishe = Goldfyshe {
        swim_speed: 998,
        name: "alec_guinness".to_string(),
    };
    let arc = Arc::new(vec![
        Box::new(catte)  as Box<dyn Pet+Sync+Send>,
        Box::new(dogge1) as Box<dyn Pet+Sync+Send>,
        Box::new(fishe)  as Box<dyn Pet+Sync+Send>,
        Box::new(dogge2) as Box<dyn Pet+Sync+Send>]);
    let (tx1, rx1) = channel();
    let arc1 = arc.clone();
    let t1 = thread::spawn(move|| { check_legs(arc1); tx1.send(()); });
    let (tx2, rx2) = channel();
    let arc2 = arc.clone();
    let t2 = thread::spawn(move|| { check_names(arc2); tx2.send(()); });
    let (tx3, rx3) = channel();
    let arc3 = arc.clone();
    let t3 = thread::spawn(move|| { check_pedigree(arc3); tx3.send(()); });
    rx1.recv();
    rx2.recv();
    rx3.recv();
    t1.join();
    t2.join();
    t3.join();
}

fn check_legs(arc: Arc<Vec<Box<dyn Pet+Sync+Send>>>) {
    let mut legs = 0;
    for pet in arc.iter() {
        legs += pet.num_legs();
    }
    assert!(legs == 12);
}
fn check_names(arc: Arc<Vec<Box<dyn Pet+Sync+Send>>>) {
    for pet in arc.iter() {
        pet.name(Box::new(|name| {
            assert!(name.as_bytes()[0] == 'a' as u8 && name.as_bytes()[1] == 'l' as u8);
        }))
    }
}
fn check_pedigree(arc: Arc<Vec<Box<dyn Pet+Sync+Send>>>) {
    for pet in arc.iter() {
        assert!(pet.of_good_pedigree());
    }
}
