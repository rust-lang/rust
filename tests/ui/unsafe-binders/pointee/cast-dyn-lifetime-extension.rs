//@ compile-flags: -Znext-solver
//@ known-bug: unknown

// Casting out of an unsafe binder should require an `unwrap_binder!` call..

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::fmt::Debug;
use std::mem::ManuallyDrop as MD;

fn direct<'x>(p: *const MD<dyn Debug + 'x>) -> *const MD<dyn Debug + 'static> {
    p as _
}

fn through_binder<'x>(p: *const MD<dyn Debug + 'x>) -> *const MD<dyn Debug + 'static> {
    let q = p as *const unsafe<'a> MD<dyn Debug + 'a>;
    // FIXME(unsafe_binders): this should be an error, like in `direct`.
    q as _
}

fn main() {}
