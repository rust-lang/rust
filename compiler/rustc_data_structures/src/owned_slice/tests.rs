use std::{cell::Cell, ops::Deref, rc::Rc};

use crate::{
    owned_slice::{slice_owned, try_slice_owned, OwnedSlice},
    OnDrop,
};

#[test]
fn smoke() {
    let slice = slice_owned(vec![1, 2, 3, 4, 5, 6], Vec::as_slice);

    assert_eq!(&*slice, [1, 2, 3, 4, 5, 6]);
}

#[test]
fn static_storage() {
    let slice = slice_owned(Box::new(String::from("what")), |_| b"bytes boo");

    assert_eq!(&*slice, b"bytes boo");
}

#[test]
fn slice_the_slice() {
    let slice = slice_owned(vec![1, 2, 3, 4, 5, 6], Vec::as_slice);
    let slice = slice_owned(slice, |s| &s[1..][..4]);
    let slice = slice_owned(slice, |s| s);
    let slice = slice_owned(slice, |s| &s[1..]);

    assert_eq!(&*slice, &[1, 2, 3, 4, 5, 6][1..][..4][1..]);
}

#[test]
fn try_and_fail() {
    let res = try_slice_owned(vec![0], |v| v.get(12..).ok_or(()));

    assert!(res.is_err());
}

#[test]
fn boxed() {
    // It's important that we don't cause UB because of `Box`'es uniqueness

    let boxed: Box<[u8]> = vec![1, 1, 2, 3, 5, 8, 13, 21].into_boxed_slice();
    let slice = slice_owned(boxed, Deref::deref);

    assert_eq!(&*slice, [1, 1, 2, 3, 5, 8, 13, 21]);
}

#[test]
fn drop() {
    let flag = Rc::new(Cell::new(false));
    let flag_prime = Rc::clone(&flag);
    let d = OnDrop(move || flag_prime.set(true));

    let slice = slice_owned(d, |_| &[]);

    assert_eq!(flag.get(), false);

    std::mem::drop(slice);

    assert_eq!(flag.get(), true);
}

#[test]
fn send_sync() {
    crate::sync::assert_send::<OwnedSlice>();
    crate::sync::assert_sync::<OwnedSlice>();
}
