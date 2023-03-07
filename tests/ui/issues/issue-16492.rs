// run-pass
#![allow(non_snake_case)]

use std::rc::Rc;
use std::cell::Cell;

struct Field {
    number: usize,
    state: Rc<Cell<usize>>
}

impl Field {
    fn new(number: usize, state: Rc<Cell<usize>>) -> Field {
        Field {
            number: number,
            state: state
        }
    }
}

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
