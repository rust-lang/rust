//@ run-pass

#![allow(dead_code)]

use std::mem::needs_drop;
use std::mem::ManuallyDrop;

struct NeedDrop;

impl Drop for NeedDrop {
    fn drop(&mut self) {}
}

// Constant expressios allow `NoDrop` to go out of scope,
// unlike a value of the interior type implementing `Drop`.
static X: () = (NoDrop { inner: ManuallyDrop::new(NeedDrop) }, ()).1;

const Y: () = (NoDrop { inner: ManuallyDrop::new(NeedDrop) }, ()).1;

const fn _f() {
    (NoDrop { inner: ManuallyDrop::new(NeedDrop) }, ()).1
}

// A union that scrubs the drop glue from its inner type
union NoDrop<T> {
    inner: ManuallyDrop<T>,
}

// Copy currently can't be implemented on drop-containing unions,
// this may change later
// https://github.com/rust-lang/rust/pull/38934#issuecomment-271219289

// // We should be able to implement Copy for NoDrop
// impl<T> Copy for NoDrop<T> {}
// impl<T> Clone for NoDrop<T> {fn clone(&self) -> Self { *self }}

// // We should be able to implement Copy for things using NoDrop
// #[derive(Copy, Clone)]
struct Foo {
    x: NoDrop<Box<u8>>,
}

struct Baz {
    x: NoDrop<Box<u8>>,
    y: Box<u8>,
}

union ActuallyDrop<T> {
    inner: ManuallyDrop<T>,
}

impl<T> Drop for ActuallyDrop<T> {
    fn drop(&mut self) {}
}

fn main() {
    // NoDrop should not make needs_drop true
    assert!(!needs_drop::<Foo>());
    assert!(!needs_drop::<NoDrop<u8>>());
    assert!(!needs_drop::<NoDrop<Box<u8>>>());
    // presence of other drop types should still work
    assert!(needs_drop::<Baz>());
    // drop impl on union itself should work
    assert!(needs_drop::<ActuallyDrop<u8>>());
    assert!(needs_drop::<ActuallyDrop<Box<u8>>>());
}
