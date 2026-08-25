//@ check-pass

#![feature(unsafe_cell_access)]
#![feature(sync_unsafe_cell)]
#![feature(once_cell_try_insert)]
#![feature(once_cell_try)]

use std::cell::{Cell, RefCell, SyncUnsafeCell, UnsafeCell};
use std::cell::{LazyCell, OnceCell};
use std::ops::Deref;

fn lazy_cell() {
    const A: LazyCell<i32> = LazyCell::new(|| 0);

    let _ = LazyCell::force(&A);
    //~^ WARN mutation of an interior mutable `const` item with call to `force`
}

fn once_cell() {
    const A: OnceCell<i32> = OnceCell::new();

    let _ = A.set(10);
    //~^ WARN mutation of an interior mutable `const` item with call to `set`

    let _ = A.try_insert(20);
    //~^ WARN mutation of an interior mutable `const` item with call to `try_insert`

    let _ = A.get_or_init(|| 30);
    //~^ WARN mutation of an interior mutable `const` item with call to `get_or_init`

    let _ = A.get_or_try_init(|| Ok::<_, ()>(40));
    //~^ WARN mutation of an interior mutable `const` item with call to `get_or_try_init`
}

fn cell() {
    const A: Cell<i32> = Cell::new(0);

    let _ = A.set(1);
    //~^ WARN mutation of an interior mutable `const` item with call to `set`

    let _ = A.swap(&A);
    //~^ WARN mutation of an interior mutable `const` item with call to `swap`

    let _ = A.replace(2);
    //~^ WARN mutation of an interior mutable `const` item with call to `replace`

    let _ = A.get();
    //~^ WARN mutation of an interior mutable `const` item with call to `get`

    let _ = A.update(|x| x + 1);
    //~^ WARN mutation of an interior mutable `const` item with call to `update`
}

fn ref_cell() {
    const A: RefCell<i32> = RefCell::new(0);

    let _ = A.replace(1);
    //~^ WARN mutation of an interior mutable `const` item with call to `replace`

    let _ = A.replace_with(|x| *x + 2);
    //~^ WARN mutation of an interior mutable `const` item with call to `replace_with`

    let _ = A.swap(&A);
    //~^ WARN mutation of an interior mutable `const` item with call to `swap`

    let _ = A.borrow();
    //~^ WARN mutation of an interior mutable `const` item with call to `borrow`

    let _ = A.try_borrow();
    //~^ WARN mutation of an interior mutable `const` item with call to `try_borrow`

    let _ = A.borrow_mut();
    //~^ WARN mutation of an interior mutable `const` item with call to `borrow_mut`

    let _ = A.try_borrow_mut();
    //~^ WARN mutation of an interior mutable `const` item with call to `try_borrow_mut`
}

fn unsafe_cell() {
    const A: UnsafeCell<i32> = UnsafeCell::new(0);

    let _ = unsafe { A.replace(1) };
    //~^ WARN mutation of an interior mutable `const` item with call to `replace`

    let _ = A.get();
    //~^ WARN mutation of an interior mutable `const` item with call to `get`

    unsafe {
        let _ = A.as_ref_unchecked();
        //~^ WARN mutation of an interior mutable `const` item with call to `as_ref_unchecked`

        let _ = A.as_mut_unchecked();
        //~^ WARN mutation of an interior mutable `const` item with call to `as_mut_unchecked`
    }
}

fn sync_unsafe_cell() {
    const A: SyncUnsafeCell<i32> = SyncUnsafeCell::new(0);

    let _ = A.get();
    //~^ WARN mutation of an interior mutable `const` item with call to `get`
}

fn main() {}
