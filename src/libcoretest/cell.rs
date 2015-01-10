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
    let x = Cell::new(10i);
    assert!(x == Cell::new(10));
    assert!(x.get() == 10);
    x.set(20);
    assert!(x == Cell::new(20));
    assert!(x.get() == 20);

    let y = Cell::new((30i, 40i));
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
    let x = RefCell::new(0i);
    let _b1 = x.borrow();
    x.borrow();
}

#[test]
fn no_mut_then_imm_borrow() {
    let x = RefCell::new(0i);
    let _b1 = x.borrow_mut();
    assert!(x.try_borrow().is_none());
}

#[test]
fn no_imm_then_borrow_mut() {
    let x = RefCell::new(0i);
    let _b1 = x.borrow();
    assert!(x.try_borrow_mut().is_none());
}

#[test]
fn no_double_borrow_mut() {
    let x = RefCell::new(0i);
    let _b1 = x.borrow_mut();
    assert!(x.try_borrow_mut().is_none());
}

#[test]
fn imm_release_borrow_mut() {
    let x = RefCell::new(0i);
    {
        let _b1 = x.borrow();
    }
    x.borrow_mut();
}

#[test]
fn mut_release_borrow_mut() {
    let x = RefCell::new(0i);
    {
        let _b1 = x.borrow_mut();
    }
    x.borrow();
}

#[test]
fn double_borrow_single_release_no_borrow_mut() {
    let x = RefCell::new(0i);
    let _b1 = x.borrow();
    {
        let _b2 = x.borrow();
    }
    assert!(x.try_borrow_mut().is_none());
}

#[test]
#[should_fail]
fn discard_doesnt_unborrow() {
    let x = RefCell::new(0i);
    let _b = x.borrow();
    let _ = _b;
    let _b = x.borrow_mut();
}

#[test]
#[allow(unstable)]
fn clone_ref_updates_flag() {
    let x = RefCell::new(0i);
    {
        let b1 = x.borrow();
        assert!(x.try_borrow_mut().is_none());
        {
            let _b2 = clone_ref(&b1);
            assert!(x.try_borrow_mut().is_none());
        }
        assert!(x.try_borrow_mut().is_none());
    }
    assert!(x.try_borrow_mut().is_some());
}

#[test]
fn as_unsafe_cell() {
    let c1: Cell<uint> = Cell::new(0u);
    c1.set(1u);
    assert_eq!(1u, unsafe { *c1.as_unsafe_cell().get() });

    let c2: Cell<uint> = Cell::new(0u);
    unsafe { *c2.as_unsafe_cell().get() = 1u; }
    assert_eq!(1u, c2.get());

    let r1: RefCell<uint> = RefCell::new(0u);
    *r1.borrow_mut() = 1u;
    assert_eq!(1u, unsafe { *r1.as_unsafe_cell().get() });

    let r2: RefCell<uint> = RefCell::new(0u);
    unsafe { *r2.as_unsafe_cell().get() = 1u; }
    assert_eq!(1u, *r2.borrow());
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
