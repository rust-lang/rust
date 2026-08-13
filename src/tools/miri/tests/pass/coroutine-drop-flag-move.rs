//! Moving an `Unpin` coroutine must not typed-move saved locals whose drop
//! flags are false. Those locals are stored as `(T, bool)` rather than
//! `Option<T>`, so the field may be moved-from or never initialized.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/161026>.
//@revisions: stack tree tree_implicit_writes
//@compile-flags: -Zmiri-ignore-leaks
//@[tree]compile-flags: -Zmiri-tree-borrows -Zmiri-ignore-leaks
//@[tree_implicit_writes]compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes -Zmiri-ignore-leaks
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

struct DropU8 {
    _r: u8,
}
impl Drop for DropU8 {
    fn drop(&mut self) {}
}

struct DropMut<T: 'static>(&'static mut T);
impl<T: 'static> Drop for DropMut<T> {
    fn drop(&mut self) {}
}

fn uninit_drop_local() {
    let mut a = #[coroutine]
    || {
        let _b: DropU8;
        if false {
            _b = DropU8 { _r: 1 };
        }
        yield 4;
    };
    assert!(matches!(Pin::new(&mut a).resume(()), CoroutineState::Yielded(4)));
    // Move while the drop flag for `_b` is false. This used to typed-move
    // uninitialized memory as `u8`.
    let _moved = a;
}

fn moved_from_ref_local() {
    let mut a = #[coroutine]
    || {
        let b = DropMut(Box::leak(Box::new(1)));
        let c;
        if true {
            c = b;
        } else {
            c = DropMut(Box::leak(Box::new(2)));
        }
        *c.0 = 3;
        yield 4;
        *c.0 = 5;
    };
    assert!(matches!(Pin::new(&mut a).resume(()), CoroutineState::Yielded(4)));
    // Move while `b`'s drop flag is false. Typed-moving `b` used to retag the
    // reference that was moved into `c`, freezing it under Tree Borrows.
    let mut d = a;
    assert!(matches!(Pin::new(&mut d).resume(()), CoroutineState::Complete(())));
}

fn main() {
    uninit_drop_local();
    moved_from_ref_local();
}
