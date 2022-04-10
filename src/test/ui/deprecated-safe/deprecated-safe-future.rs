// aux-build:deprecated-safe.rs
// check-pass
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(type_alias_impl_trait)]
#![warn(unused_unsafe)]

extern crate deprecated_safe;

use deprecated_safe::depr_safe_future;

trait Bla {
    type T;
}

type Tait = impl FnOnce();

impl Bla for u32 {
    type T = Tait;
}

fn foo3() -> Tait {
    depr_safe_future
}

fn foo4() -> impl FnOnce() {
    depr_safe_future
}

unsafe fn unsafe_fn() {
    depr_safe_future();
}

fn main() {
    // test for dyn Fn() coercion where arguments match (no args)
    let fn_impl: Box<dyn Fn()> = Box::new(depr_safe_future);
    let fn_impl: Box<dyn FnMut()> = Box::new(depr_safe_future);
    let fn_impl: Box<dyn FnOnce()> = Box::new(depr_safe_future);

    // test for dyn Fn() coercion where arguments don't match, where an
    // intermediate fn() will be used
    let fn_impl: Box<dyn Fn()> = Box::new(depr_safe_future);
    let fn_impl: Box<dyn FnMut()> = Box::new(depr_safe_future);
    let fn_impl: Box<dyn FnOnce()> = Box::new(depr_safe_future);

    // test variant where coercion occurs on a function argument instead of a variable
    fn_taking_dyn_fn_impl(Box::new(depr_safe_future));
    fn_taking_dyn_fnmut_impl(Box::new(depr_safe_future));
    fn_taking_dyn_fnonce_impl(Box::new(depr_safe_future));

    // test for non-dyn Fn() coercion (no unsizing)
    fn_taking_fn_impl(depr_safe_future);
    fn_taking_fnmut_impl(depr_safe_future);
    fn_taking_fnonce_impl(depr_safe_future);

    // usage without unsafe should lint
    depr_safe_future();
    depr_safe_future();

    // test for fn() coercion where arguments match (no args)
    let depr_safe_generic_fn_ptr: fn() = depr_safe_future;

    // test for fn() coercion where arguments don't match, where an
    // intermediate fn() will be used
    let depr_safe_generic_fn_ptr: fn() = depr_safe_future;
    let depr_safe_generic_fn_impl: Box<dyn Fn()> = Box::new(depr_safe_future);
    let mut depr_safe_generic_fnmut_impl: Box<dyn FnMut()> = Box::new(depr_safe_future);
    let depr_safe_generic_fnonce_impl: Box<dyn FnOnce()> = Box::new(depr_safe_future);

    // these shouldn't lint, appropriate unsafe usage
    let unsafe_depr_safe_generic_fn_ptr: unsafe fn() = depr_safe_future;
    fn_taking_unsafe_fn_ptr(depr_safe_future);
    unsafe {
        //~^ WARN unnecessary `unsafe` block
        depr_safe_future();
    }

    // all of these coercions should lint
    fn_taking_fn_ptr(depr_safe_future);
    fn_taking_dyn_fn_impl(Box::new(depr_safe_future));
    fn_taking_dyn_fnmut_impl(Box::new(depr_safe_future));
    fn_taking_dyn_fnonce_impl(Box::new(depr_safe_future));
    fn_taking_fn_impl(depr_safe_future);
    fn_taking_fnmut_impl(depr_safe_future);
    fn_taking_fnonce_impl(depr_safe_future);
}

fn fn_taking_fn_ptr(_: fn()) {}
fn fn_taking_unsafe_fn_ptr(_: unsafe fn()) {}

fn fn_taking_dyn_fn_impl(_: Box<dyn Fn()>) {}
fn fn_taking_dyn_fnmut_impl(_: Box<dyn FnMut()>) {}
fn fn_taking_dyn_fnonce_impl(_: Box<dyn FnOnce()>) {}

fn fn_taking_fn_impl(_: impl Fn()) {}
fn fn_taking_fnmut_impl(_: impl FnMut()) {}
fn fn_taking_fnonce_impl(_: impl FnOnce()) {}
