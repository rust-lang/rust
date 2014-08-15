// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty

#![feature(unsafe_destructor)]

use std::rc::Rc;
use std::cell::Cell;

struct Field {
    number: uint,
    state: Rc<Cell<uint>>
}

impl Field {
    fn new(number: uint, state: Rc<Cell<uint>>) -> Field {
        Field {
            number: number,
            state: state
        }
    }
}

#[unsafe_destructor] // because Field isn't Send
impl Drop for Field {
    fn drop(&mut self) {
        println!("Dropping field {}", self.number);
        assert_eq!(self.state.get(), self.number);
        self.state.set(self.state.get()+1);
    }
}

struct NoDropImpl {
    _one: Field,
    _two: Field,
    _three: Field
}

struct HasDropImpl {
    _one: Field,
    _two: Field,
    _three: Field
}

#[unsafe_destructor] // because HasDropImpl isn't Send
impl Drop for HasDropImpl {
    fn drop(&mut self) {
        println!("HasDropImpl.drop()");
        assert_eq!(self._one.state.get(), 0);
        self._one.state.set(1);
    }
}

pub fn main() {
    let state = Rc::new(Cell::new(1));
    let noImpl = NoDropImpl {
        _one: Field::new(1, state.clone()),
        _two: Field::new(2, state.clone()),
        _three: Field::new(3, state.clone())
    };
    drop(noImpl);
    assert_eq!(state.get(), 4);

    state.set(0);
    let hasImpl = HasDropImpl {
        _one: Field::new(1, state.clone()),
        _two: Field::new(2, state.clone()),
        _three: Field::new(3, state.clone())
    };
    drop(hasImpl);
    assert_eq!(state.get(), 4);
}
