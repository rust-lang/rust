// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::cell::*;
use core::default::Default;
use std::mem::drop;

#[test]
fn smoketest_cell() {
    let x = Cell::new(10);
    assert!(x == Cell::new(10));
    assert!(x.get() == 10);
    x.set(20);
    assert!(x == Cell::new(20));
    assert!(x.get() == 20);

    let y = Cell::new((30, 40));
    assert!(y == Cell::new((30, 40)));
    assert!(y.get() == (30, 40));
}

#[test]
fn cell_has_sensible_show() {
    let x = Cell::new("foo bar");
    assert!(format!("{:?}", x).contains(x.get()));

    x.set("baz qux");
    assert!(format!("{:?}", x).contains(x.get()));
}

#[test]
fn ref_and_refmut_have_sensible_show() {
    let refcell = RefCell::new("foo");

    let refcell_refmut = refcell.borrow_mut();
    assert!(format!("{:?}", refcell_refmut).contains("foo"));
    drop(refcell_refmut);

    let refcell_ref = refcell.borrow();
    assert!(format!("{:?}", refcell_ref).contains("foo"));
    drop(refcell_ref);
}

#[test]
fn double_imm_borrow() {
    let x = RefCell::new(0);
    let _b1 = x.borrow();
    x.borrow();
}

#[test]
fn no_mut_then_imm_borrow() {
    let x = RefCell::new(0);
    let _b1 = x.borrow_mut();
    assert_eq!(x.borrow_state(), BorrowState::Writing);
}

#[test]
fn no_imm_then_borrow_mut() {
    let x = RefCell::new(0);
    let _b1 = x.borrow();
    assert_eq!(x.borrow_state(), BorrowState::Reading);
}

#[test]
fn no_double_borrow_mut() {
    let x = RefCell::new(0);
    assert_eq!(x.borrow_state(), BorrowState::Unused);
    let _b1 = x.borrow_mut();
    assert_eq!(x.borrow_state(), BorrowState::Writing);
}

#[test]
fn imm_release_borrow_mut() {
    let x = RefCell::new(0);
    {
        let _b1 = x.borrow();
    }
    x.borrow_mut();
}

#[test]
fn mut_release_borrow_mut() {
    let x = RefCell::new(0);
    {
        let _b1 = x.borrow_mut();
    }
    x.borrow();
}

#[test]
fn double_borrow_single_release_no_borrow_mut() {
    let x = RefCell::new(0);
    let _b1 = x.borrow();
    {
        let _b2 = x.borrow();
    }
    assert_eq!(x.borrow_state(), BorrowState::Reading);
}

#[test]
#[should_panic]
fn discard_doesnt_unborrow() {
    let x = RefCell::new(0);
    let _b = x.borrow();
    let _ = _b;
    let _b = x.borrow_mut();
}

#[test]
fn clone_ref_updates_flag() {
    let x = RefCell::new(0);
    {
        let b1 = x.borrow();
        assert_eq!(x.borrow_state(), BorrowState::Reading);
        {
            let _b2 = clone_ref(&b1);
            assert_eq!(x.borrow_state(), BorrowState::Reading);
        }
        assert_eq!(x.borrow_state(), BorrowState::Reading);
    }
    assert_eq!(x.borrow_state(), BorrowState::Unused);
}

#[test]
fn as_unsafe_cell() {
    let c1: Cell<usize> = Cell::new(0);
    c1.set(1);
    assert_eq!(1, unsafe { *c1.as_unsafe_cell().get() });

    let c2: Cell<usize> = Cell::new(0);
    unsafe { *c2.as_unsafe_cell().get() = 1; }
    assert_eq!(1, c2.get());

    let r1: RefCell<usize> = RefCell::new(0);
    *r1.borrow_mut() = 1;
    assert_eq!(1, unsafe { *r1.as_unsafe_cell().get() });

    let r2: RefCell<usize> = RefCell::new(0);
    unsafe { *r2.as_unsafe_cell().get() = 1; }
    assert_eq!(1, *r2.borrow());
}

#[test]
fn cell_default() {
    let cell: Cell<u32> = Default::default();
    assert_eq!(0, cell.get());
}

#[test]
fn refcell_default() {
    let cell: RefCell<u64> = Default::default();
    assert_eq!(0, *cell.borrow());
}
