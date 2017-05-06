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
    assert!(x.try_borrow().is_err());
}

#[test]
fn no_imm_then_borrow_mut() {
    let x = RefCell::new(0);
    let _b1 = x.borrow();
    assert!(x.try_borrow_mut().is_err());
}

#[test]
fn no_double_borrow_mut() {
    let x = RefCell::new(0);
    assert!(x.try_borrow().is_ok());
    let _b1 = x.borrow_mut();
    assert!(x.try_borrow().is_err());
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
    assert!(x.try_borrow().is_ok());
    assert!(x.try_borrow_mut().is_err());
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
fn ref_clone_updates_flag() {
    let x = RefCell::new(0);
    {
        let b1 = x.borrow();
        assert!(x.try_borrow().is_ok());
        assert!(x.try_borrow_mut().is_err());
        {
            let _b2 = Ref::clone(&b1);
            assert!(x.try_borrow().is_ok());
            assert!(x.try_borrow_mut().is_err());
        }
        assert!(x.try_borrow().is_ok());
        assert!(x.try_borrow_mut().is_err());
    }
    assert!(x.try_borrow().is_ok());
    assert!(x.try_borrow_mut().is_ok());
}

#[test]
fn ref_map_does_not_update_flag() {
    let x = RefCell::new(Some(5));
    {
        let b1: Ref<Option<u32>> = x.borrow();
        assert!(x.try_borrow().is_ok());
        assert!(x.try_borrow_mut().is_err());
        {
            let b2: Ref<u32> = Ref::map(b1, |o| o.as_ref().unwrap());
            assert_eq!(*b2, 5);
            assert!(x.try_borrow().is_ok());
            assert!(x.try_borrow_mut().is_err());
        }
        assert!(x.try_borrow().is_ok());
        assert!(x.try_borrow_mut().is_ok());
    }
    assert!(x.try_borrow().is_ok());
    assert!(x.try_borrow_mut().is_ok());
}

#[test]
fn ref_map_accessor() {
    struct X(RefCell<(u32, char)>);
    impl X {
        fn accessor(&self) -> Ref<u32> {
            Ref::map(self.0.borrow(), |tuple| &tuple.0)
        }
    }
    let x = X(RefCell::new((7, 'z')));
    let d: Ref<u32> = x.accessor();
    assert_eq!(*d, 7);
}

#[test]
fn ref_mut_map_accessor() {
    struct X(RefCell<(u32, char)>);
    impl X {
        fn accessor(&self) -> RefMut<u32> {
            RefMut::map(self.0.borrow_mut(), |tuple| &mut tuple.0)
        }
    }
    let x = X(RefCell::new((7, 'z')));
    {
        let mut d: RefMut<u32> = x.accessor();
        assert_eq!(*d, 7);
        *d += 1;
    }
    assert_eq!(*x.0.borrow(), (8, 'z'));
}

#[test]
fn as_ptr() {
    let c1: Cell<usize> = Cell::new(0);
    c1.set(1);
    assert_eq!(1, unsafe { *c1.as_ptr() });

    let c2: Cell<usize> = Cell::new(0);
    unsafe { *c2.as_ptr() = 1; }
    assert_eq!(1, c2.get());

    let r1: RefCell<usize> = RefCell::new(0);
    *r1.borrow_mut() = 1;
    assert_eq!(1, unsafe { *r1.as_ptr() });

    let r2: RefCell<usize> = RefCell::new(0);
    unsafe { *r2.as_ptr() = 1; }
    assert_eq!(1, *r2.borrow());
}

#[test]
fn cell_default() {
    let cell: Cell<u32> = Default::default();
    assert_eq!(0, cell.get());
}

#[test]
fn cell_set() {
    let cell = Cell::new(10);
    cell.set(20);
    assert_eq!(20, cell.get());

    let cell = Cell::new("Hello".to_owned());
    cell.set("World".to_owned());
    assert_eq!("World".to_owned(), cell.into_inner());
}

#[test]
fn cell_replace() {
    let cell = Cell::new(10);
    assert_eq!(10, cell.replace(20));
    assert_eq!(20, cell.get());

    let cell = Cell::new("Hello".to_owned());
    assert_eq!("Hello".to_owned(), cell.replace("World".to_owned()));
    assert_eq!("World".to_owned(), cell.into_inner());
}

#[test]
fn cell_into_inner() {
    let cell = Cell::new(10);
    assert_eq!(10, cell.into_inner());

    let cell = Cell::new("Hello world".to_owned());
    assert_eq!("Hello world".to_owned(), cell.into_inner());
}

#[test]
fn refcell_default() {
    let cell: RefCell<u64> = Default::default();
    assert_eq!(0, *cell.borrow());
}

#[test]
fn unsafe_cell_unsized() {
    let cell: &UnsafeCell<[i32]> = &UnsafeCell::new([1, 2, 3]);
    {
        let val: &mut [i32] = unsafe { &mut *cell.get() };
        val[0] = 4;
        val[2] = 5;
    }
    let comp: &mut [i32] = &mut [4, 2, 5];
    assert_eq!(unsafe { &mut *cell.get() }, comp);
}

#[test]
fn refcell_unsized() {
    let cell: &RefCell<[i32]> = &RefCell::new([1, 2, 3]);
    {
        let b = &mut *cell.borrow_mut();
        b[0] = 4;
        b[2] = 5;
    }
    let comp: &mut [i32] = &mut [4, 2, 5];
    assert_eq!(&*cell.borrow(), comp);
}

#[test]
fn refcell_ref_coercion() {
    let cell: RefCell<[i32; 3]> = RefCell::new([1, 2, 3]);
    {
        let mut cellref: RefMut<[i32; 3]> = cell.borrow_mut();
        cellref[0] = 4;
        let mut coerced: RefMut<[i32]> = cellref;
        coerced[2] = 5;
    }
    {
        let comp: &mut [i32] = &mut [4, 2, 5];
        let cellref: Ref<[i32; 3]> = cell.borrow();
        assert_eq!(&*cellref, comp);
        let coerced: Ref<[i32]> = cellref;
        assert_eq!(&*coerced, comp);
    }
}
