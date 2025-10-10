//@ known-bug: #107975
//@ compile-flags: -Copt-level=2
//@ run-pass
//@ ignore-backends: gcc

// https://github.com/rust-lang/rust/issues/107975#issuecomment-1431758601

use std::{
    cell::{Ref, RefCell},
    ptr,
};

fn main() {
    let a: usize = {
        let v = 0u8;
        ptr::from_ref(&v).expose_provenance()
    };
    let b: usize = {
        let v = 0u8;
        ptr::from_ref(&v).expose_provenance()
    };
    let i: usize = b - a;

    // A surprise tool that will help us later.
    let arr = [
        RefCell::new(Some(Box::new(1u8))),
        RefCell::new(None),
        RefCell::new(None),
        RefCell::new(None),
    ];

    // `i` is not 0
    assert_ne!(i, 0);

    // Let's borrow the `i`-th element.
    // If `i` is out of bounds, indexing will panic.
    let r: Ref<Option<Box<u8>>> = arr[i].borrow();

    // If we got here, it means `i` was in bounds.
    // Now, two options are possible:
    // EITHER `i` is not 0 (as we have asserted above),
    // so the unwrap will panic, because only the 0-th element is `Some`
    // OR the assert lied, `i` *is* 0, and the `unwrap` will not panic.
    let r: &Box<u8> = r.as_ref().unwrap();

    // If we got here, it means `i` *was* actually 0.
    // Let's ignore the fact that the assert has lied
    // and try to take a mutable reference to the 0-th element.
    // `borrow_mut` should panic, because we are sill holding on
    // to a shared `Ref` for the same `RefCell`.
    *arr[0].borrow_mut() = None;

    // But it doesn't panic!
    // We have successfully replaced `Some(Box)` with `None`,
    // while holding a shared reference to it.
    // No unsafe involved.

    // The `Box` has been deallocated by now, so this is a dangling reference!
    let r: &u8 = &*r;
    println!("{:p}", r);
    println!("{}", i);

    // The following might segfault. Or it might not.
    // Depends on the platform semantics
    // and whatever happened to the pointed-to memory after deallocation.
    // let u: u8 = *r;
    // println!("{u}");
}
