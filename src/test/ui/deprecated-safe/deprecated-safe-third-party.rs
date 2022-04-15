// check-pass
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(deprecated_safe)]
#![feature(type_alias_impl_trait)]
#![warn(unused_unsafe)]

use std::ffi::{OsStr, OsString};

#[deprecated_safe(since = "1.61.0", note = "reason")]
unsafe fn set_var<K: AsRef<OsStr>, V: AsRef<OsStr>>(key: K, value: V) {
    std::env::set_var(key, value)
}

#[deprecated_safe(since = "1.61.0", note = "reason")]
unsafe trait PreviouslySafeTrait {}

#[deprecated_safe(since = "99.99.99", note = "reason")]
unsafe fn set_var_future<K: AsRef<OsStr>, V: AsRef<OsStr>>(key: K, value: V) {
    std::env::set_var(key, value)
}

#[deprecated_safe(since = "99.99.99", note = "reason")]
unsafe trait PreviouslySafeTraitFuture {}

#[deprecated_safe(since = "TBD", note = "reason")]
unsafe fn set_var_tbd<K: AsRef<OsStr>, V: AsRef<OsStr>>(key: K, value: V) {
    std::env::set_var(key, value)
}

#[deprecated_safe(since = "TBD", note = "reason")]
unsafe trait PreviouslySafeTraitTbd {}

trait Bla {
    type T;
}

type Tait = impl FnOnce(OsString, OsString); //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
//~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
type TaitTbd = impl FnOnce(OsString, OsString); //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
//~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
type TaitFuture = impl FnOnce(OsString, OsString); //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
//~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function

impl Bla for u32 {
    type T = Tait;
}

fn foo3() -> Tait {
    set_var //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
}

fn foo4() -> impl FnOnce(OsString, OsString) { //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    set_var //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
}

fn foo5() -> TaitFuture {
    set_var_future //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
}

fn foo6() -> impl FnOnce(OsString, OsString) { //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    set_var_future //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
}

fn foo7() -> TaitTbd {
    set_var_tbd //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
}

fn foo8() -> impl FnOnce(OsString, OsString) { //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    set_var_tbd //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
}

struct PreviouslySafeTraitImpl;
impl PreviouslySafeTrait for PreviouslySafeTraitImpl {} //~ WARN use of trait `PreviouslySafeTrait` without an `unsafe impl` declaration has been deprecated as it is now an unsafe trait
impl PreviouslySafeTraitFuture for PreviouslySafeTraitImpl {} //~ WARN use of trait `PreviouslySafeTraitFuture` without an `unsafe impl` declaration has been deprecated as it is now an unsafe trait
impl PreviouslySafeTraitTbd for PreviouslySafeTraitImpl {} //~ WARN use of trait `PreviouslySafeTraitTbd` without an `unsafe impl` declaration has been deprecated as it is now an unsafe trait

fn main() {
    let fn_impl: Box<dyn Fn(OsString, OsString)> = Box::new(set_var); //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
                                                                      //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
                                                                      //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnMut(OsString, OsString)> = Box::new(set_var); //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
                                                                         //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnOnce(OsString, OsString)> = Box::new(set_var); //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
                                                                          //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function

    let fn_impl: Box<dyn Fn(OsString, OsString)> = Box::new(set_var_future); //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
                                                                                 //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
                                                                                 //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnMut(OsString, OsString)> = Box::new(set_var_future); //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
                                                                                //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnOnce(OsString, OsString)> = Box::new(set_var_future); //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
                                                                                 //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function

    let fn_impl: Box<dyn Fn(OsString, OsString)> = Box::new(set_var_tbd); //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
                                                                          //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
                                                                          //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnMut(OsString, OsString)> = Box::new(set_var_tbd); //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
                                                                             //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnOnce(OsString, OsString)> = Box::new(set_var_tbd); //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
                                                                              //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    // test for non-dyn Fn() coercion (no unsizing)
    fn_taking_fn_impl(set_var); //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl(set_var); //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl(set_var); //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function

    fn_taking_fn_impl(set_var_future); //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl(set_var_future); //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl(set_var_future); //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function

    fn_taking_fn_impl(set_var_tbd); //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl(set_var_tbd); //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl(set_var_tbd); //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function

    // usage without unsafe should lint
    set_var("TEST_DEPRECATED_SAFE", "set_var_safe"); //~ WARN use of function `set_var` without an unsafe function or block has been deprecated as it is now an unsafe function
    set_var("TEST_DEPRECATED_SAFE", "set_var_safe"); //~ WARN use of function `set_var` without an unsafe function or block has been deprecated as it is now an unsafe function

    set_var_future("TEST_DEPRECATED_SAFE", "set_var_safe"); //~ WARN use of function `set_var_future` without an unsafe function or block has been deprecated as it is now an unsafe function
    set_var_future("TEST_DEPRECATED_SAFE", "set_var_safe"); //~ WARN use of function `set_var_future` without an unsafe function or block has been deprecated as it is now an unsafe function

    set_var_tbd("TEST_DEPRECATED_SAFE", "set_var_safe"); //~ WARN use of function `set_var_tbd` without an unsafe function or block has been deprecated as it is now an unsafe function
    set_var_tbd("TEST_DEPRECATED_SAFE", "set_var_safe"); //~ WARN use of function `set_var_tbd` without an unsafe function or block has been deprecated as it is now an unsafe function

    let set_var_fn_ptr: fn(OsString, OsString) = set_var; //~ WARN use of function `set_var` as a normal fn pointer has been deprecated as it is now an unsafe function
    let set_var_fn_ptr: fn(OsString, OsString) = set_var_future; //~ WARN use of function `set_var_future` as a normal fn pointer has been deprecated as it is now an unsafe function
    let set_var_fn_ptr: fn(OsString, OsString) = set_var_tbd; //~ WARN use of function `set_var_tbd` as a normal fn pointer has been deprecated as it is now an unsafe function

    // these shouldn't lint, appropriate unsafe usage
    let unsafe_set_var_fn_ptr: unsafe fn(OsString, OsString) = set_var;
    let unsafe_set_var_fn_ptr: unsafe fn(OsString, OsString) = set_var_future;
    let unsafe_set_var_fn_ptr: unsafe fn(OsString, OsString) = set_var_tbd;
    fn_taking_unsafe_fn_ptr(set_var);
    fn_taking_unsafe_fn_ptr(set_var_future);
    fn_taking_unsafe_fn_ptr(set_var_tbd);
    unsafe {
        set_var("TEST_DEPRECATED_SAFE", "set_var_unsafe");
        set_var_future("TEST_DEPRECATED_SAFE", "set_var_unsafe");
        set_var_tbd("TEST_DEPRECATED_SAFE", "set_var_unsafe");
    }

    // all of these coercions should lint
    fn_taking_fn_ptr(set_var); //~ WARN use of function `set_var` as a normal fn pointer has been deprecated as it is now an unsafe function
    fn_taking_dyn_fn_impl(Box::new(set_var)); //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
                                                  //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl(Box::new(set_var)); //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
                                                  //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl(Box::new(set_var)); //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
                                                  //~| WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fn_impl(set_var); //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl(set_var); //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl(set_var); //~ WARN use of function `set_var` as a closure has been deprecated as it is now an unsafe function


    fn_taking_fn_ptr(set_var_future); //~ WARN use of function `set_var_future` as a normal fn pointer has been deprecated as it is now an unsafe function
    fn_taking_dyn_fn_impl(Box::new(set_var_future)); //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
                                              //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl(Box::new(set_var_future)); //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
                                                 //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl(Box::new(set_var_future)); //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
                                                  //~| WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fn_impl(set_var_future); //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl(set_var_future); //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl(set_var_future); //~ WARN use of function `set_var_future` as a closure has been deprecated as it is now an unsafe function

    fn_taking_fn_ptr(set_var_tbd); //~ WARN use of function `set_var_tbd` as a normal fn pointer has been deprecated as it is now an unsafe function
    fn_taking_dyn_fn_impl(Box::new(set_var_tbd)); //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
                                                  //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl(Box::new(set_var_tbd)); //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
                                                  //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl(Box::new(set_var_tbd)); //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
                                                  //~| WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fn_impl(set_var_tbd); //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl(set_var_tbd); //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl(set_var_tbd); //~ WARN use of function `set_var_tbd` as a closure has been deprecated as it is now an unsafe function
}

fn fn_taking_fn_ptr(_: fn(OsString, OsString)) {}
fn fn_taking_unsafe_fn_ptr(_: unsafe fn(OsString, OsString)) {}

fn fn_taking_dyn_fn_impl(_: Box<dyn Fn(OsString, OsString)>) {}
fn fn_taking_dyn_fnmut_impl(_: Box<dyn FnMut(OsString, OsString)>) {}
fn fn_taking_dyn_fnonce_impl(_: Box<dyn FnOnce(OsString, OsString)>) {}

fn fn_taking_fn_impl(_: impl Fn(OsString, OsString)) {}
fn fn_taking_fnmut_impl(_: impl FnMut(OsString, OsString)) {}
fn fn_taking_fnonce_impl(_: impl FnOnce(OsString, OsString)) {}
