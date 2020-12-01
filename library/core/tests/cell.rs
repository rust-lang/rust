use core::cell::*;
use core::default::Default;
use std::mem::drop;

#[test]
fn smoketest_cell() {
    let x = Cell::new(10);
    assert_eq!(x, Cell::new(10));
    assert_eq!(x.get(), 10);
    x.set(20);
    assert_eq!(x, Cell::new(20));
    assert_eq!(x.get(), 20);

    let y = Cell::new((30, 40));
    assert_eq!(y, Cell::new((30, 40)));
    assert_eq!(y.get(), (30, 40));
}

#[test]
fn cell_update() {
    let x = Cell::new(10);

    assert_eq!(x.update(|x| x + 5), 15);
    assert_eq!(x.get(), 15);

    assert_eq!(x.update(|x| x / 3), 5);
    assert_eq!(x.get(), 5);
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
        let b1: Ref<'_, Option<u32>> = x.borrow();
        assert!(x.try_borrow().is_ok());
        assert!(x.try_borrow_mut().is_err());
        {
            let b2: Ref<'_, u32> = Ref::map(b1, |o| o.as_ref().unwrap());
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
fn ref_map_split_updates_flag() {
    let x = RefCell::new([1, 2]);
    {
        let b1 = x.borrow();
        assert!(x.try_borrow().is_ok());
        assert!(x.try_borrow_mut().is_err());
        {
            let (_b2, _b3) = Ref::map_split(b1, |slc| slc.split_at(1));
            assert!(x.try_borrow().is_ok());
            assert!(x.try_borrow_mut().is_err());
        }
        assert!(x.try_borrow().is_ok());
        assert!(x.try_borrow_mut().is_ok());
    }
    assert!(x.try_borrow().is_ok());
    assert!(x.try_borrow_mut().is_ok());

    {
        let b1 = x.borrow_mut();
        assert!(x.try_borrow().is_err());
        assert!(x.try_borrow_mut().is_err());
        {
            let (_b2, _b3) = RefMut::map_split(b1, |slc| slc.split_at_mut(1));
            assert!(x.try_borrow().is_err());
            assert!(x.try_borrow_mut().is_err());
            drop(_b2);
            assert!(x.try_borrow().is_err());
            assert!(x.try_borrow_mut().is_err());
        }
        assert!(x.try_borrow().is_ok());
        assert!(x.try_borrow_mut().is_ok());
    }
    assert!(x.try_borrow().is_ok());
    assert!(x.try_borrow_mut().is_ok());
}

#[test]
fn ref_map_split() {
    let x = RefCell::new([1, 2]);
    let (b1, b2) = Ref::map_split(x.borrow(), |slc| slc.split_at(1));
    assert_eq!(*b1, [1]);
    assert_eq!(*b2, [2]);
}

#[test]
fn ref_mut_map_split() {
    let x = RefCell::new([1, 2]);
    {
        let (mut b1, mut b2) = RefMut::map_split(x.borrow_mut(), |slc| slc.split_at_mut(1));
        assert_eq!(*b1, [1]);
        assert_eq!(*b2, [2]);
        b1[0] = 2;
        b2[0] = 1;
    }
    assert_eq!(*x.borrow(), [2, 1]);
}

#[test]
fn ref_map_accessor() {
    struct X(RefCell<(u32, char)>);
    impl X {
        fn accessor(&self) -> Ref<'_, u32> {
            Ref::map(self.0.borrow(), |tuple| &tuple.0)
        }
    }
    let x = X(RefCell::new((7, 'z')));
    let d: Ref<'_, u32> = x.accessor();
    assert_eq!(*d, 7);
}

#[test]
fn ref_mut_map_accessor() {
    struct X(RefCell<(u32, char)>);
    impl X {
        fn accessor(&self) -> RefMut<'_, u32> {
            RefMut::map(self.0.borrow_mut(), |tuple| &mut tuple.0)
        }
    }
    let x = X(RefCell::new((7, 'z')));
    {
        let mut d: RefMut<'_, u32> = x.accessor();
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
    unsafe {
        *c2.as_ptr() = 1;
    }
    assert_eq!(1, c2.get());

    let r1: RefCell<usize> = RefCell::new(0);
    *r1.borrow_mut() = 1;
    assert_eq!(1, unsafe { *r1.as_ptr() });

    let r2: RefCell<usize> = RefCell::new(0);
    unsafe {
        *r2.as_ptr() = 1;
    }
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
fn cell_exterior() {
    #[derive(Copy, Clone)]
    #[allow(dead_code)]
    struct Point {
        x: isize,
        y: isize,
        z: isize,
    }

    fn f(p: &Cell<Point>) {
        assert_eq!(p.get().z, 12);
        p.set(Point { x: 10, y: 11, z: 13 });
        assert_eq!(p.get().z, 13);
    }

    let a = Point { x: 10, y: 11, z: 12 };
    let b = &Cell::new(a);
    assert_eq!(b.get().z, 12);
    f(b);
    assert_eq!(a.z, 12);
    assert_eq!(b.get().z, 13);
}

#[test]
fn cell_does_not_clone() {
    #[derive(Copy)]
    #[allow(dead_code)]
    struct Foo {
        x: isize,
    }

    impl Clone for Foo {
        fn clone(&self) -> Foo {
            // Using Cell in any way should never cause clone() to be
            // invoked -- after all, that would permit evil user code to
            // abuse `Cell` and trigger crashes.

            panic!();
        }
    }

    let x = Cell::new(Foo { x: 22 });
    let _y = x.get();
    let _z = x.clone();
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
        let mut cellref: RefMut<'_, [i32; 3]> = cell.borrow_mut();
        cellref[0] = 4;
        let mut coerced: RefMut<'_, [i32]> = cellref;
        coerced[2] = 5;
    }
    {
        let comp: &mut [i32] = &mut [4, 2, 5];
        let cellref: Ref<'_, [i32; 3]> = cell.borrow();
        assert_eq!(&*cellref, comp);
        let coerced: Ref<'_, [i32]> = cellref;
        assert_eq!(&*coerced, comp);
    }
}

#[test]
#[should_panic]
fn refcell_swap_borrows() {
    let x = RefCell::new(0);
    let _b = x.borrow();
    let y = RefCell::new(1);
    x.swap(&y);
}

#[test]
#[should_panic]
fn refcell_replace_borrows() {
    let x = RefCell::new(0);
    let _b = x.borrow();
    x.replace(1);
}

#[test]
fn refcell_format() {
    let name = RefCell::new("rust");
    let what = RefCell::new("rocks");
    let msg = format!("{name} {}", &*what.borrow(), name = &*name.borrow());
    assert_eq!(msg, "rust rocks".to_string());
}

#[allow(dead_code)]
fn const_cells() {
    const UNSAFE_CELL: UnsafeCell<i32> = UnsafeCell::new(3);
    const _: i32 = UNSAFE_CELL.into_inner();

    const REF_CELL: RefCell<i32> = RefCell::new(3);
    const _: i32 = REF_CELL.into_inner();

    const CELL: Cell<i32> = Cell::new(3);
    const _: i32 = CELL.into_inner();
}
