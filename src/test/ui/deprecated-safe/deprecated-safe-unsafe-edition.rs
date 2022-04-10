// aux-build:deprecated-safe.rs
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(deprecated_safe)]
#![warn(unused_unsafe)]

extern crate deprecated_safe;

use deprecated_safe::{depr_safe_2015, depr_safe_2018};
use std::ffi::{OsStr, OsString};

unsafe fn unsafe_fn() {
    depr_safe_2015();
    depr_safe_2018();
}

fn main() {
    // test for dyn Fn() coercion where arguments don't match, where an
    // intermediate fn() will be used
    let fn_impl: Box<dyn Fn()> = Box::new(depr_safe_2015); //~ ERROR expected a `Fn<()>` closure, found `unsafe fn() {depr_safe_2015}`
    let fn_impl: Box<dyn FnMut()> = Box::new(depr_safe_2015); //~ ERROR expected a `FnMut<()>` closure, found `unsafe fn() {depr_safe_2015}`
    let fn_impl: Box<dyn FnOnce()> = Box::new(depr_safe_2015); //~ ERROR expected a `FnOnce<()>` closure, found `unsafe fn() {depr_safe_2015}`

    // test variant where coercion occurs on a function argument instead of a variable
    fn_taking_dyn_fn_impl(Box::new(depr_safe_2015)); //~ ERROR expected a `Fn<()>` closure, found `unsafe fn() {depr_safe_2015}`
    fn_taking_dyn_fnmut_impl(Box::new(depr_safe_2015)); //~ ERROR expected a `FnMut<()>` closure, found `unsafe fn() {depr_safe_2015}`
    fn_taking_dyn_fnonce_impl(Box::new(depr_safe_2015)); //~ ERROR expected a `FnOnce<()>` closure, found `unsafe fn() {depr_safe_2015}`

    // test for non-dyn Fn() coercion (no unsizing)
    fn_taking_fn_impl(depr_safe_2015); //~ ERROR expected a `Fn<()>` closure, found `unsafe fn() {depr_safe_2015}`
    fn_taking_fnmut_impl(depr_safe_2015); //~ ERROR expected a `FnMut<()>` closure, found `unsafe fn() {depr_safe_2015}`
    fn_taking_fnonce_impl(depr_safe_2015); //~ ERROR expected a `FnOnce<()>` closure, found `unsafe fn() {depr_safe_2015}`

    // test for fn() coercion where arguments don't match, where an
    // intermediate fn() will be used
    let depr_safe_fn_ptr: fn() = depr_safe_2015; //~ ERROR mismatched types
    let depr_safe_fn_impl: Box<dyn Fn()> = Box::new(depr_safe_2015); //~ ERROR expected a `Fn<()>` closure, found `unsafe fn() {depr_safe_2015}`
    let mut depr_safe_fnmut_impl: Box<dyn FnMut()> = Box::new(depr_safe_2015); //~ ERROR expected a `FnMut<()>` closure, found `unsafe fn() {depr_safe_2015}`
    let depr_safe_fnonce_impl: Box<dyn FnOnce()> = Box::new(depr_safe_2015); //~ ERROR expected a `FnOnce<()>` closure, found `unsafe fn() {depr_safe_2015}`

    // these shouldn't lint, appropriate unsafe usage
    let unsafe_depr_safe_fn_ptr: unsafe fn() = depr_safe_2015;
    fn_taking_unsafe_fn_ptr(depr_safe_2015);
    unsafe {
        depr_safe_2015();
    }

    // all of these coercions should lint
    fn_taking_fn_ptr(depr_safe_2015); //~ ERROR mismatched types
    fn_taking_dyn_fn_impl(Box::new(depr_safe_2015)); //~ ERROR expected a `Fn<()>` closure, found `unsafe fn() {depr_safe_2015}`
    fn_taking_dyn_fnmut_impl(Box::new(depr_safe_2015)); //~ ERROR expected a `FnMut<()>` closure, found `unsafe fn() {depr_safe_2015}`
    fn_taking_dyn_fnonce_impl(Box::new(depr_safe_2015)); //~ ERROR expected a `FnOnce<()>` closure, found `unsafe fn() {depr_safe_2015}`
    fn_taking_fn_impl(depr_safe_2015); //~ ERROR expected a `Fn<()>` closure, found `unsafe fn() {depr_safe_2015}`
    fn_taking_fnmut_impl(depr_safe_2015); //~ ERROR expected a `FnMut<()>` closure, found `unsafe fn() {depr_safe_2015}`
    fn_taking_fnonce_impl(depr_safe_2015); //~ ERROR expected a `FnOnce<()>` closure, found `unsafe fn() {depr_safe_2015}`
}

fn main_2018() {
    // test for dyn Fn() coercion where arguments don't match, where an
    // intermediate fn() will be used
    let fn_impl: Box<dyn Fn()> = Box::new(depr_safe_2018); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
                                                           //~| WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
                                                           //~| WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnMut()> = Box::new(depr_safe_2018); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
                                                              //~| WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnOnce()> = Box::new(depr_safe_2018); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
                                                               //~| WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function

    // test variant where coercion occurs on a function argument instead of a variable
    fn_taking_dyn_fn_impl(Box::new(depr_safe_2018)); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl(Box::new(depr_safe_2018)); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl(Box::new(depr_safe_2018)); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function

    // test for non-dyn Fn() coercion (no unsizing)
    fn_taking_fn_impl(depr_safe_2018); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl(depr_safe_2018); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl(depr_safe_2018); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function

    // test for fn() coercion where arguments don't match, where an
    // intermediate fn() will be used
    let depr_safe_fn_ptr: fn() = depr_safe_2018; //~ WARN use of function `deprecated_safe::depr_safe_2018` as a normal fn pointer has been deprecated as it is now an unsafe function
    let depr_safe_fn_impl: Box<dyn Fn()> = Box::new(depr_safe_2018); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
    let mut depr_safe_fnmut_impl: Box<dyn FnMut()> = Box::new(depr_safe_2018); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
    let depr_safe_fnonce_impl: Box<dyn FnOnce()> = Box::new(depr_safe_2018); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function

    // these shouldn't lint, appropriate unsafe usage
    let unsafe_depr_safe_fn_ptr: unsafe fn() = depr_safe_2018;
    fn_taking_unsafe_fn_ptr(depr_safe_2018);
    unsafe {
        depr_safe_2018();
    }

    // all of these coercions should lint
    fn_taking_fn_ptr(depr_safe_2018); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a normal fn pointer has been deprecated as it is now an unsafe function
    fn_taking_dyn_fn_impl(Box::new(depr_safe_2018)); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl(Box::new(depr_safe_2018)); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl(Box::new(depr_safe_2018)); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fn_impl(depr_safe_2018); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl(depr_safe_2018); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl(depr_safe_2018); //~ WARN use of function `deprecated_safe::depr_safe_2018` as a closure has been deprecated as it is now an unsafe function
}

fn fn_taking_fn_ptr(_: fn()) {}
fn fn_taking_unsafe_fn_ptr(_: unsafe fn()) {}

fn fn_taking_dyn_fn_impl(_: Box<dyn Fn()>) {}
fn fn_taking_dyn_fnmut_impl(_: Box<dyn FnMut()>) {}
fn fn_taking_dyn_fnonce_impl(_: Box<dyn FnOnce()>) {}

fn fn_taking_fn_impl(_: impl Fn()) {}
fn fn_taking_fnmut_impl(_: impl FnMut()) {}
fn fn_taking_fnonce_impl(_: impl FnOnce()) {}
