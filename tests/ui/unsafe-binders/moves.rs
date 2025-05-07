#![feature(unsafe_binders)]
//~^ WARN the feature `unsafe_binders` is incomplete

use std::mem::{ManuallyDrop, drop};
use std::unsafe_binder::{unwrap_binder, wrap_binder};

#[derive(Default)]
struct NotCopyInner;
type NotCopy = ManuallyDrop<NotCopyInner>;

fn use_after_wrap() {
    unsafe {
        let base = NotCopy::default();
        let binder: unsafe<> NotCopy = wrap_binder!(base);
        drop(base);
        //~^ ERROR use of moved value: `base`
    }
}

fn move_out_of_wrap() {
    unsafe {
        let binder: unsafe<> NotCopy = wrap_binder!(NotCopy::default());
        drop(unwrap_binder!(binder));
        drop(unwrap_binder!(binder));
        //~^ ERROR use of moved value: `binder`
    }
}

fn not_conflicting() {
    unsafe {
        let binder: unsafe<> (NotCopy, NotCopy) =
            wrap_binder!((NotCopy::default(), NotCopy::default()));
        drop(unwrap_binder!(binder).0);
        drop(unwrap_binder!(binder).1);
        // ^ NOT a problem, since the moves are disjoint.
        drop(unwrap_binder!(binder).0);
        //~^ ERROR use of moved value: `binder.0`
    }
}

fn main() {}
