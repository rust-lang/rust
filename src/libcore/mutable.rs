// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Module for wrapping freezable data structures in managed boxes.
Normally freezable data structures require an unaliased reference,
such as `T` or `~T`, so that the compiler can track when they are
being mutated.  The `managed<T>` type converts these static checks into
dynamic checks: your program will fail if you attempt to perform
mutation when the data structure should be immutable.

*/

use util::with;
use cast::transmute_immut;

enum Mode { ReadOnly, Mutable, Immutable }

struct Data<T> {
    priv mut value: T,
    priv mut mode: Mode
}

pub type Mut<T> = Data<T>;

pub fn Mut<T>(t: T) -> Mut<T> {
    Data {value: t, mode: ReadOnly}
}

pub fn unwrap<T>(m: Mut<T>) -> T {
    // Borrowck should prevent us from calling unwrap while the value
    // is in use, as that would be a move from a borrowed value.
    assert!((m.mode as uint) == (ReadOnly as uint));
    let Data {value: value, mode: _} = m;
    value
}

pub impl<T> Data<T> {
    fn borrow_mut<R>(&self, op: &fn(t: &mut T) -> R) -> R {
        match self.mode {
            Immutable => fail!(~"currently immutable"),
            ReadOnly | Mutable => {}
        }

        do with(&mut self.mode, Mutable) {
            op(&mut self.value)
        }
    }

    fn borrow_const<R>(&self, op: &fn(t: &const T) -> R) -> R {
        op(&const self.value)
    }

    fn borrow_imm<R>(&self, op: &fn(t: &T) -> R) -> R {
        match self.mode {
          Mutable => fail!(~"currently mutable"),
          ReadOnly | Immutable => {}
        }

        do with(&mut self.mode, Immutable) {
            op(unsafe{transmute_immut(&mut self.value)})
        }
    }

    #[inline(always)]
    fn unwrap(self) -> T { unwrap(self) }
}

#[test]
#[ignore(cfg(windows))]
#[should_fail]
pub fn test_mut_in_imm() {
    let m = @Mut(1);
    do m.borrow_imm |_p| {
        do m.borrow_mut |_q| {
            // should not be permitted
        }
    }
}

#[test]
#[ignore(cfg(windows))]
#[should_fail]
pub fn test_imm_in_mut() {
    let m = @Mut(1);
    do m.borrow_mut |_p| {
        do m.borrow_imm |_q| {
            // should not be permitted
        }
    }
}

#[test]
pub fn test_const_in_mut() {
    let m = @Mut(1);
    do m.borrow_mut |p| {
        do m.borrow_const |q| {
            assert!(*p == *q);
            *p += 1;
            assert!(*p == *q);
        }
    }
}

#[test]
pub fn test_mut_in_const() {
    let m = @Mut(1);
    do m.borrow_const |p| {
        do m.borrow_mut |q| {
            assert!(*p == *q);
            *q += 1;
            assert!(*p == *q);
        }
    }
}

#[test]
pub fn test_imm_in_const() {
    let m = @Mut(1);
    do m.borrow_const |p| {
        do m.borrow_imm |q| {
            assert!(*p == *q);
        }
    }
}

#[test]
pub fn test_const_in_imm() {
    let m = @Mut(1);
    do m.borrow_imm |p| {
        do m.borrow_const |q| {
            assert!(*p == *q);
        }
    }
}


#[test]
#[ignore(cfg(windows))]
#[should_fail]
pub fn test_mut_in_imm_in_const() {
    let m = @Mut(1);
    do m.borrow_const |_p| {
        do m.borrow_imm |_q| {
            do m.borrow_mut |_r| {
            }
        }
    }
}

