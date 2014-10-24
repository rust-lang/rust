// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:the right fail

#![feature(unsafe_destructor)]

use std::cell::Cell;
use std::kinds::marker;
use std::mem;
use std::ops::{Placer,PlacementAgent};

#[deriving(PartialEq, Eq, Show)]
enum Entry {
    DropInterim(uint),
    FinalizeInterim(uint),
    DropAtom(&'static str),
}

type Record = Vec<Entry>;

struct Atom {
    record: *mut Record,
    name: &'static str,
}

struct InterimAtom {
    record: *mut Record,
    name_todo: &'static str,
    my_count: uint,
}

struct AtomPool {
    record: *mut Record,
    total_count: uint,
}

struct ExpectDropRecord<'a> {
    record: *mut Record,
    expect: &'a [Entry],
}

#[unsafe_destructor]
impl<'a> Drop for ExpectDropRecord<'a> {
    fn drop(&mut self) {
        unsafe {
            assert_eq!((*self.record).as_slice(), self.expect);
        }
    }
}

pub fn main() {
    let mut record = vec![];
    let record = &mut record as *mut Record;
    let mut pool = &mut AtomPool::new(record);

    let expect = [FinalizeInterim(0),
                  DropInterim(1),
                  DropAtom("hello")];
    let expect = ExpectDropRecord { record: record, expect: expect };
    inner(pool);
}

fn inner(mut pool: &mut AtomPool) {
    let a = box (pool) "hello";
    let b = box (pool) { fail!("the right fail"); "world" };
    let c = box (pool) { fail!("we never"); "get here " };
}

impl AtomPool {
    fn new(record: *mut Record) -> AtomPool {
        AtomPool { record: record, total_count: 0 }
    }
}

impl<'a> Placer<&'static str, Atom, InterimAtom> for &'a mut AtomPool {
    fn make_place(&mut self) -> InterimAtom {
        let c = self.total_count;
        self.total_count = c + 1;
        InterimAtom {
            record: self.record,
            name_todo: "",
            my_count: c,
        }
    }
}

impl PlacementAgent<&'static str, Atom> for InterimAtom {
    unsafe fn pointer(&self) -> *mut &'static str {
        mem::transmute(&self.name_todo)
    }
    unsafe fn finalize(self) -> Atom {
        unsafe { (*self.record).push(FinalizeInterim(self.my_count)) }
        let ret = Atom {
            record: self.record,
            name: self.name_todo,
        };
        mem::forget(self);
        ret
    }
}

impl Drop for InterimAtom {
    fn drop(&mut self) {
        unsafe { (*self.record).push(DropInterim(self.my_count)) }
    }
}

impl Drop for Atom {
    fn drop(&mut self) {
        unsafe { (*self.record).push(DropAtom(self.name)) }
    }
}
