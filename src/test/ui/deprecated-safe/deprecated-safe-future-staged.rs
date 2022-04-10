// aux-build:deprecated-safe.rs
// check-pass
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(type_alias_impl_trait)]
#![feature(staged_api)]
#![stable(feature = "deprecated-safe-test", since = "1.61.0")]
#![warn(deprecated_safe_in_future, unused_unsafe)]

extern crate deprecated_safe;

use deprecated_safe::{depr_safe_future};

trait Bla {
    type T;
}

type Tait = impl FnOnce(); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
//~| WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function

impl Bla for u32 {
    type T = Tait;
}

fn foo3() -> Tait {
    depr_safe_future //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
}

fn foo4() -> impl FnOnce() {
    //~^ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    depr_safe_future //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
}

unsafe fn unsafe_fn() {
    depr_safe_future();
}

fn main() {
    // test for dyn Fn() coercion where arguments match (no args)
    let fn_impl: Box<dyn Fn()> = Box::new(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
                                                             //~| WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
                                                             //~| WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnMut()> = Box::new(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
                                                                //~| WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnOnce()> = Box::new(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
                                                                 //~| WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function

    // test for dyn Fn() coercion where arguments don't match, where an
    // intermediate fn() will be used
    let fn_impl: Box<dyn Fn()> = Box::new(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnMut()> = Box::new(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnOnce()> = Box::new(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function

    // test variant where coercion occurs on a function argument instead of a variable
    fn_taking_dyn_fn_impl(Box::new(depr_safe_future)); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl(Box::new(depr_safe_future)); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl(Box::new(depr_safe_future)); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function

    // test for non-dyn Fn() coercion (no unsizing)
    fn_taking_fn_impl(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function

    // usage without unsafe should lint
    depr_safe_future(); //~ WARN use of function `deprecated_safe::depr_safe_future` without an unsafe function or block has been deprecated as it is now an unsafe function
    depr_safe_future(); //~ WARN use of function `deprecated_safe::depr_safe_future` without an unsafe function or block has been deprecated as it is now an unsafe function

    // test for fn() coercion where arguments match (no args)
    let depr_safe_generic_fn_ptr: fn() = depr_safe_future; //~ WARN use of function `deprecated_safe::depr_safe_future` as a normal fn pointer has been deprecated as it is now an unsafe function

    // test for fn() coercion where arguments don't match, where an
    // intermediate fn() will be used
    let depr_safe_generic_fn_ptr: fn() = depr_safe_future; //~ WARN use of function `deprecated_safe::depr_safe_future` as a normal fn pointer has been deprecated as it is now an unsafe function
    let depr_safe_generic_fn_impl: Box<dyn Fn()> = Box::new(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    let mut depr_safe_generic_fnmut_impl: Box<dyn FnMut()> = Box::new(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    let depr_safe_generic_fnonce_impl: Box<dyn FnOnce()> = Box::new(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function

    // these shouldn't lint, appropriate unsafe usage
    let unsafe_depr_safe_generic_fn_ptr: unsafe fn() = depr_safe_future;
    fn_taking_unsafe_fn_ptr(depr_safe_future);
    unsafe {
        depr_safe_future();
    }

    // all of these coercions should lint
    fn_taking_fn_ptr(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a normal fn pointer has been deprecated as it is now an unsafe function
    fn_taking_dyn_fn_impl(Box::new(depr_safe_future)); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl(Box::new(depr_safe_future)); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl(Box::new(depr_safe_future)); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fn_impl(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl(depr_safe_future); //~ WARN use of function `deprecated_safe::depr_safe_future` as a closure has been deprecated as it is now an unsafe function
}

fn fn_taking_fn_ptr(_: fn()) {}
fn fn_taking_unsafe_fn_ptr(_: unsafe fn()) {}

fn fn_taking_dyn_fn_impl(_: Box<dyn Fn()>) {}
fn fn_taking_dyn_fnmut_impl(_: Box<dyn FnMut()>) {}
fn fn_taking_dyn_fnonce_impl(_: Box<dyn FnOnce()>) {}

fn fn_taking_fn_impl(_: impl Fn()) {}
fn fn_taking_fnmut_impl(_: impl FnMut()) {}
fn fn_taking_fnonce_impl(_: impl FnOnce()) {}
