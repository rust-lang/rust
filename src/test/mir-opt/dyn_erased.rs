// compile-flags: -Zinline-mir=no

#![feature(rustc_attrs)]

use std::fmt::Display;
use std::marker::PhantomData;

#[inline(never)]
fn generic<T: Display>(a: &T) {
    println!("{}", a)
}

#[inline(never)]
fn non_generic(a: u32) {
    println!("{}", a)
}

// EMIT_MIR dyn_erased.normal.DynErased.after.mir
// EMIT_MIR dyn_erased.normal.DynErasedBody.after.mir
#[rustc_dyn]
fn normal<T: Display>(f: &T) {
    generic(f);
    generic(&"foo"); // Check this remaine a direct call.
    non_generic(5); // Check this is a direct call.
}

// EMIT_MIR dyn_erased.coreturn.DynErased.after.mir
// EMIT_MIR dyn_erased.coreturn.DynErasedBody.after.mir
#[rustc_dyn]
fn coreturn<T: Display>(f: &T) -> &T {
    generic(f);
    non_generic(5); // Check this is a direct call.
    f // Check we transmute this return type for LLVM.
}

// EMIT_MIR dyn_erased.good_niche.DynErased.after.mir
// EMIT_MIR dyn_erased.good_niche.DynErasedBody.after.mir
#[rustc_dyn]
fn good_niche<T>(f: Option<&T>) -> &T {
    if let Some(f) = f { f } else { panic!() }
}

trait Foo {
    const C: usize;
}

impl Foo for u32 {
    const C: usize = 13;
}

// EMIT_MIR dyn_erased.assoc_const.DynErased.after.mir
// EMIT_MIR dyn_erased.assoc_const.DynErasedBody.after.mir
#[rustc_dyn]
fn assoc_const<T: Foo>(_: &T) -> usize {
    T::C
}

struct Droppy<T>(PhantomData<T>);

impl<T> Drop for Droppy<T> {
    #[inline(never)]
    fn drop(&mut self) {
        println!("drop!");
    }
}

// EMIT_MIR dyn_erased.dropping.DynErased.after.mir
// EMIT_MIR dyn_erased.dropping.DynErasedBody.after.mir
#[rustc_dyn]
fn dropping<T: Foo>(_: &T) -> () {
    let _: Droppy<T> = Droppy(PhantomData);
}

// Use a main function to verify that monomorphizations are collected correctly.
fn main() {
    normal(&42);
    assert!(coreturn(&42) == &42);
    good_niche(Some(&42));
    assoc_const(&42);
    dropping(&42);
}
