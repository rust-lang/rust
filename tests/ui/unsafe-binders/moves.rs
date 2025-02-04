//@ known-bug: unknown

#![feature(unsafe_binders)]
// FIXME(unsafe_binders) ~^ WARN the feature `unsafe_binders` is incomplete

use std::unsafe_binder::{wrap_binder, unwrap_binder};
use std::mem::{drop, ManuallyDrop};

struct NotCopyInner;
type NotCopy = ManuallyDrop<NotCopyInner>;

fn use_after_wrap() {
    unsafe {
        let base = NotCopy;
        let binder: unsafe<> NotCopy = wrap_binder!(base);
        drop(base);
        // FIXME(unsafe_binders) ~^ ERROR use of moved value: `base`
    }
}

fn move_out_of_wrap() {
    unsafe {
        let binder: unsafe<> NotCopy = wrap_binder!(NotCopy);
        drop(unwrap_binder!(binder));
        drop(unwrap_binder!(binder));
        // FIXME(unsafe_binders) ~^ ERROR use of moved value: `binder`
    }
}

fn not_conflicting() {
    unsafe {
        let binder: unsafe<> (NotCopy, NotCopy) = wrap_binder!((NotCopy, NotCopy));
        drop(unwrap_binder!(binder).0);
        drop(unwrap_binder!(binder).1);
        // ^ NOT a problem.
        drop(unwrap_binder!(binder).0);
        // FIXME(unsafe_binders) ~^ ERROR use of moved value: `binder.0`
    }
}

fn main() {}
