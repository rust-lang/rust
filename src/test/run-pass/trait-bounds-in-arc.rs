// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that a heterogeneous list of existential types can be put inside an Arc
// and shared between tasks as long as all types fulfill Freeze+Send.

// xfail-fast

extern mod extra;
use extra::arc;
use std::comm;
use std::task;
use std::cell;

trait Pet {
    fn name(&self, blk: &fn(&str));
    fn num_legs(&self) -> uint;
    fn of_good_pedigree(&self) -> bool;
}

struct Catte {
    num_whiskers: uint,
    name: ~str,
}

struct Dogge {
    bark_decibels: uint,
    tricks_known: uint,
    name: ~str,
}

struct Goldfyshe {
    swim_speed: uint,
    name: ~str,
}

impl Pet for Catte {
    fn name(&self, blk: &fn(&str)) { blk(self.name) }
    fn num_legs(&self) -> uint { 4 }
    fn of_good_pedigree(&self) -> bool { self.num_whiskers >= 4 }
}
impl Pet for Dogge {
    fn name(&self, blk: &fn(&str)) { blk(self.name) }
    fn num_legs(&self) -> uint { 4 }
    fn of_good_pedigree(&self) -> bool {
        self.bark_decibels < 70 || self.tricks_known > 20
    }
}
impl Pet for Goldfyshe {
    fn name(&self, blk: &fn(&str)) { blk(self.name) }
    fn num_legs(&self) -> uint { 0 }
    fn of_good_pedigree(&self) -> bool { self.swim_speed >= 500 }
}

fn main() {
    let catte = Catte { num_whiskers: 7, name: ~"alonzo_church" };
    let dogge1 = Dogge { bark_decibels: 100, tricks_known: 42, name: ~"alan_turing" };
    let dogge2 = Dogge { bark_decibels: 55,  tricks_known: 11, name: ~"albert_einstein" };
    let fishe = Goldfyshe { swim_speed: 998, name: ~"alec_guinness" };
    let arc = arc::Arc::new(~[~catte  as ~Pet:Freeze+Send,
                         ~dogge1 as ~Pet:Freeze+Send,
                         ~fishe  as ~Pet:Freeze+Send,
                         ~dogge2 as ~Pet:Freeze+Send]);
    let (p1,c1) = comm::stream();
    let arc1 = cell::Cell::new(arc.clone());
    do task::spawn { check_legs(arc1.take()); c1.send(()); }
    let (p2,c2) = comm::stream();
    let arc2 = cell::Cell::new(arc.clone());
    do task::spawn { check_names(arc2.take()); c2.send(()); }
    let (p3,c3) = comm::stream();
    let arc3 = cell::Cell::new(arc.clone());
    do task::spawn { check_pedigree(arc3.take()); c3.send(()); }
    p1.recv();
    p2.recv();
    p3.recv();
}

fn check_legs(arc: arc::Arc<~[~Pet:Freeze+Send]>) {
    let mut legs = 0;
    for arc.get().iter().advance |pet| {
        legs += pet.num_legs();
    }
    assert!(legs == 12);
}
fn check_names(arc: arc::Arc<~[~Pet:Freeze+Send]>) {
    for arc.get().iter().advance |pet| {
        do pet.name |name| {
            assert!(name[0] == 'a' as u8 && name[1] == 'l' as u8);
        }
    }
}
fn check_pedigree(arc: arc::Arc<~[~Pet:Freeze+Send]>) {
    for arc.get().iter().advance |pet| {
        assert!(pet.of_good_pedigree());
    }
}
