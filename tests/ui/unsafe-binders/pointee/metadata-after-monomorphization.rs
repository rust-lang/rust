//@ compile-flags: -Znext-solver
//@ build-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""
//@ normalize-stderr: "Normalizing .* without wrapping in a `Binder`" -> "Normalizing .. without wrapping in a `Binder`"
//@ normalize-stderr: "`ProjectionClause\(.*\)` has escaping bound vars" -> "`ProjectionClause(..)` has escaping bound vars"
//@ compile-flags: -Cdebuginfo=2

// After monomorphization, the metadata of every binder must be concrete: the
// rigid projection from `alias-tail-bound-lifetime.rs` now has a concrete tail.
// This exercises pointer layout, `PtrMetadata` codegen, `size_of_val` /
// `align_of_val` through the binder, debuginfo for wide pointers to binders, and
// unwrapping through a wide pointer. It uses legacy mangling because v0
// mangling ICEs on unsafe binders (#154367).
//
// Expected: run-pass. See the `.stderr` files for the current behavior.

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features, dead_code)]

use std::fmt::Debug;
use std::mem::{ManuallyDrop as MD, align_of_val, size_of, size_of_val};
use std::marker::PhantomData;
use std::ptr;
use std::unsafe_binder::unwrap_binder;

trait Tr {
    type Assoc<'a>: ?Sized;
}

struct Slice;
impl Tr for Slice {
    type Assoc<'a> = [&'a u8];
}

struct Dyn;
impl Tr for Dyn {
    type Assoc<'a> = dyn Debug + 'a;
}

fn generic<T: Tr>(p: *const unsafe<'a> MD<T::Assoc<'a>>) -> (usize, usize) {
    assert_eq!(size_of::<*const unsafe<'a> MD<T::Assoc<'a>>>(), 2 * size_of::<usize>());
    let q: *const unsafe<'a> MD<T::Assoc<'a>>
        = ptr::from_raw_parts(p as *const (), ptr::metadata(p));
    let r = unsafe { &*q };
    let inner: &MD<T::Assoc<'_>> = unsafe { &unwrap_binder!(*r) };
    assert_eq!(size_of_val(inner), size_of_val(r));
    (size_of_val(r), align_of_val(r))
}

// The `dyn` tail is reached through a struct that isn't `ManuallyDrop`.
struct Wd<'a, T: ?Sized>(u8, PhantomData<&'a ()>, MD<T>);

fn main() {
    let (a, b, c) = (1u8, 2u8, 3u8);
    let s: &[&u8] = &[&a, &b, &c];
    let p = s as *const [&u8] as *const MD<[&u8]> as *const unsafe<'a> MD<[&'a u8]>;
    assert_eq!(generic::<Slice>(p), (3 * size_of::<usize>(), size_of::<usize>()));

    let x = MD::new(0u64);
    let p = &x as &MD<dyn Debug> as *const MD<dyn Debug> as *const unsafe<'a> MD<dyn Debug + 'a>;
    assert_eq!(generic::<Dyn>(p), (8, 8));
    let r = unsafe { &unwrap_binder!(*p) };
    assert_eq!(format!("{:?}", &**r), "0");

    let w = Wd(1u8, PhantomData, MD::new(0u64));
    let p = &w as &Wd<'_, dyn Debug> as *const Wd<'_, dyn Debug>
        as *const unsafe<'a> MD<Wd<'a, dyn Debug + 'a>>;
    let r = unsafe { &*p };
    assert_eq!(size_of_val(r), size_of_val(&w));
    let inner: &MD<Wd<'_, dyn Debug>> = unsafe { &unwrap_binder!(*r) };
    assert_eq!(format!("{:?}", &*inner.2), "0");
}
