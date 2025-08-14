//@ build-fail

#![feature(core_intrinsics)]
#![allow(warnings)]
#![crate_type = "rlib"]

use std::intrinsics;

#[derive(Copy, Clone)]
pub struct Foo(i64);
pub type Bar = &'static Fn();
pub type Quux = [u8; 100];

pub unsafe fn test_bool_load(p: &mut bool, v: bool) {
    intrinsics::atomic_load_seqcst(p);
    //~^ ERROR `atomic_load_seqcst` intrinsic: expected basic integer type, found `bool`
}

pub unsafe fn test_bool_store(p: &mut bool, v: bool) {
    intrinsics::atomic_store_seqcst(p, v);
    //~^ ERROR `atomic_store_seqcst` intrinsic: expected basic integer type, found `bool`
}

pub unsafe fn test_bool_xchg(p: &mut bool, v: bool) {
    intrinsics::atomic_xchg_seqcst(p, v);
    //~^ ERROR `atomic_xchg_seqcst` intrinsic: expected basic integer type, found `bool`
}

pub unsafe fn test_bool_cxchg(p: &mut bool, v: bool) {
    intrinsics::atomic_cxchg_seqcst_seqcst(p, v, v);
    //~^ ERROR `atomic_cxchg_seqcst_seqcst` intrinsic: expected basic integer type, found `bool`
}

pub unsafe fn test_Foo_load(p: &mut Foo, v: Foo) {
    intrinsics::atomic_load_seqcst(p);
    //~^ ERROR `atomic_load_seqcst` intrinsic: expected basic integer type, found `Foo`
}

pub unsafe fn test_Foo_store(p: &mut Foo, v: Foo) {
    intrinsics::atomic_store_seqcst(p, v);
    //~^ ERROR `atomic_store_seqcst` intrinsic: expected basic integer type, found `Foo`
}

pub unsafe fn test_Foo_xchg(p: &mut Foo, v: Foo) {
    intrinsics::atomic_xchg_seqcst(p, v);
    //~^ ERROR `atomic_xchg_seqcst` intrinsic: expected basic integer type, found `Foo`
}

pub unsafe fn test_Foo_cxchg(p: &mut Foo, v: Foo) {
    intrinsics::atomic_cxchg_seqcst_seqcst(p, v, v);
    //~^ ERROR `atomic_cxchg_seqcst_seqcst` intrinsic: expected basic integer type, found `Foo`
}

pub unsafe fn test_Bar_load(p: &mut Bar, v: Bar) {
    intrinsics::atomic_load_seqcst(p);
    //~^ ERROR expected basic integer type, found `&dyn Fn()`
}

pub unsafe fn test_Bar_store(p: &mut Bar, v: Bar) {
    intrinsics::atomic_store_seqcst(p, v);
    //~^ ERROR expected basic integer type, found `&dyn Fn()`
}

pub unsafe fn test_Bar_xchg(p: &mut Bar, v: Bar) {
    intrinsics::atomic_xchg_seqcst(p, v);
    //~^ ERROR expected basic integer type, found `&dyn Fn()`
}

pub unsafe fn test_Bar_cxchg(p: &mut Bar, v: Bar) {
    intrinsics::atomic_cxchg_seqcst_seqcst(p, v, v);
    //~^ ERROR expected basic integer type, found `&dyn Fn()`
}

pub unsafe fn test_Quux_load(p: &mut Quux, v: Quux) {
    intrinsics::atomic_load_seqcst(p);
    //~^ ERROR `atomic_load_seqcst` intrinsic: expected basic integer type, found `[u8; 100]`
}

pub unsafe fn test_Quux_store(p: &mut Quux, v: Quux) {
    intrinsics::atomic_store_seqcst(p, v);
    //~^ ERROR `atomic_store_seqcst` intrinsic: expected basic integer type, found `[u8; 100]`
}

pub unsafe fn test_Quux_xchg(p: &mut Quux, v: Quux) {
    intrinsics::atomic_xchg_seqcst(p, v);
    //~^ ERROR `atomic_xchg_seqcst` intrinsic: expected basic integer type, found `[u8; 100]`
}

pub unsafe fn test_Quux_cxchg(p: &mut Quux, v: Quux) {
    intrinsics::atomic_cxchg_seqcst_seqcst(p, v, v);
    //~^ ERROR `atomic_cxchg_seqcst_seqcst` intrinsic: expected basic integer type, found `[u8; 100]`
}
