// aux-build:deprecated-safe.rs
// check-pass
// revisions: mir thir
// NOTE(skippy) these tests output many duplicates, so deduplicate or they become brittle to changes
// [mir]compile-flags: -Zdeduplicate-diagnostics=yes
// [thir]compile-flags: -Z thir-unsafeck -Zdeduplicate-diagnostics=yes

// FIXME(skippy) add tests in combination with #[target_feature], ensure #[deprecated_safe]
// doesn't silence these accidentally

#![feature(type_alias_impl_trait)]
#![warn(unused_unsafe)]

extern crate deprecated_safe;

use deprecated_safe::{depr_safe, depr_safe_generic, depr_safe_params, DeprSafe, DeprSafeFns};
use std::ffi::{OsStr, OsString};

trait Bla {
    type T;
}

type Tait1 = impl FnOnce(); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
type Tait2 = impl FnOnce(u32, u64); //~ WARN use of function `deprecated_safe::depr_safe_params` as a closure has been deprecated as it is now an unsafe function
type Tait3 = impl FnOnce(OsString, OsString); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
type Tait5 = Box<dyn FnOnce()>;
type Tait6 = Box<dyn FnOnce(u32, u64)>;
type Tait7 = Box<dyn FnOnce(OsString, OsString)>;
type Tait8 = Box<dyn FnOnce()>;

impl Bla for u8 {
    type T = Tait1;
}
impl Bla for u16 {
    type T = Tait2;
}
impl Bla for u32 {
    type T = Tait3;
}
impl Bla for i8 {
    type T = Tait5;
}
impl Bla for i16 {
    type T = Tait6;
}
impl Bla for i32 {
    type T = Tait7;
}
impl Bla for i64 {
    type T = Tait8;
}

fn foo1() -> Tait1 {
    depr_safe //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
}

fn foo2() -> impl FnOnce() {
    //~^ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    depr_safe //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
}

fn foo3() -> Tait2 {
    depr_safe_params //~ WARN use of function `deprecated_safe::depr_safe_params` as a closure has been deprecated as it is now an unsafe function
}

fn foo4() -> impl FnOnce(u32, u64) {
    //~^ WARN use of function `deprecated_safe::depr_safe_params` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `deprecated_safe::depr_safe_params` as a closure has been deprecated as it is now an unsafe function
    depr_safe_params //~ WARN use of function `deprecated_safe::depr_safe_params` as a closure has been deprecated as it is now an unsafe function
}

fn foo5() -> Tait3 {
    depr_safe_generic //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
}

fn foo6() -> impl FnOnce(OsString, OsString) {
    //~^ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    //~| WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    depr_safe_generic //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
}

fn foo10() -> Tait5 {
    Box::new(depr_safe) //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
}

fn foo11() -> Box<dyn FnOnce()> {
    Box::new(depr_safe) //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
}

fn foo12() -> Tait6 {
    Box::new(depr_safe_params) //~ WARN use of function `deprecated_safe::depr_safe_params` as a closure has been deprecated as it is now an unsafe function
}

fn foo13() -> Box<dyn FnOnce(u32, u64)> {
    Box::new(depr_safe_params) //~ WARN use of function `deprecated_safe::depr_safe_params` as a closure has been deprecated as it is now an unsafe function
}

fn foo14() -> Tait7 {
    Box::new(depr_safe_generic) //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
}

fn foo15() -> Box<dyn FnOnce(OsString, OsString)> {
    Box::new(depr_safe_generic) //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
}

struct DeprSafeImpl;
impl DeprSafe for DeprSafeImpl {} //~ WARN use of trait `deprecated_safe::DeprSafe` without an `unsafe impl` declaration has been deprecated as it is now an unsafe trait

struct DeprSafeUnsafeImpl;
unsafe impl DeprSafe for DeprSafeUnsafeImpl {}

unsafe fn unsafe_fn() {
    depr_safe();
    depr_safe_generic("", "");
}

struct DeprSafeFnsGood;
impl DeprSafeFns for DeprSafeFnsGood {
    unsafe fn depr_safe_fn(&self) {}

    unsafe fn depr_safe_params(&self, _: u32, _: u64) {}

    unsafe fn depr_safe_fn_generic<K: AsRef<OsStr>, V: AsRef<OsStr>>(&self, key: K, value: V) {}

    unsafe fn depr_safe_fn_2015(&self) {}

    unsafe fn depr_safe_fn_2018(&self) {}
}

struct DeprSafeFnsBad;
impl DeprSafeFns for DeprSafeFnsBad {
    fn depr_safe_fn(&self) {} //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_fn` without an `unsafe fn` declaration has been deprecated as it is now an unsafe associated function

    fn depr_safe_params(&self, _: u32, _: u64) {} //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_params` without an `unsafe fn` declaration has been deprecated as it is now an unsafe associated function

    fn depr_safe_fn_generic<K: AsRef<OsStr>, V: AsRef<OsStr>>(&self, key: K, value: V) {} //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_fn_generic` without an `unsafe fn` declaration has been deprecated as it is now an unsafe associated function

    unsafe fn depr_safe_fn_2015(&self) {}

    fn depr_safe_fn_2018(&self) {} //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_fn_2018` without an `unsafe fn` declaration has been deprecated as it is now an unsafe associated function
}

fn foo0() {
    let good = DeprSafeFnsGood;
    good.depr_safe_fn(); //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_fn` without an unsafe function or block has been deprecated as it is now an unsafe associated function
    good.depr_safe_params(0, 0); //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_params` without an unsafe function or block has been deprecated as it is now an unsafe associated function
    good.depr_safe_fn_generic("", ""); //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_fn_generic` without an unsafe function or block has been deprecated as it is now an unsafe associated function
    unsafe {
        good.depr_safe_fn_2015();
    }
    good.depr_safe_fn_2018(); //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_fn_2018` without an unsafe function or block has been deprecated as it is now an unsafe associated function

    let bad = DeprSafeFnsBad;
    bad.depr_safe_fn(); //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_fn` without an unsafe function or block has been deprecated as it is now an unsafe associated function
    bad.depr_safe_params(0, 0); //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_params` without an unsafe function or block has been deprecated as it is now an unsafe associated function
    bad.depr_safe_fn_generic("", ""); //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_fn_generic` without an unsafe function or block has been deprecated as it is now an unsafe associated function
    unsafe {
        bad.depr_safe_fn_2015();
    }
    bad.depr_safe_fn_2018(); //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_fn_2018` without an unsafe function or block has been deprecated as it is now an unsafe associated function

    unsafe {
        good.depr_safe_fn();
        good.depr_safe_params(0, 0);
        good.depr_safe_fn_generic("", "");
        good.depr_safe_fn_2015();
        good.depr_safe_fn_2018();
    }
}

fn main() {
    // test for dyn Fn() coercion where arguments match (no args)
    let fn_impl: Box<dyn Fn()> = Box::new(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnMut()> = Box::new(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnOnce()> = Box::new(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function

    // test that second usage still lints
    let fn_impl: Box<dyn Fn()> = Box::new(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnMut()> = Box::new(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnOnce()> = Box::new(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function

    // test variant where coercion occurs on a function argument instead of a variable
    fn_taking_dyn_fn_impl(Box::new(depr_safe)); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl(Box::new(depr_safe)); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl(Box::new(depr_safe)); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function

    // test that second usage still lints
    fn_taking_dyn_fn_impl(Box::new(depr_safe)); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl(Box::new(depr_safe)); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl(Box::new(depr_safe)); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function

    // test for non-dyn Fn() coercion (no unsizing)
    fn_taking_fn_impl(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function

    // test that second usage still lints
    fn_taking_fn_impl(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function

    // usage without unsafe should lint
    depr_safe(); //~ WARN use of function `deprecated_safe::depr_safe` without an unsafe function or block has been deprecated as it is now an unsafe function
    depr_safe(); //~ WARN use of function `deprecated_safe::depr_safe` without an unsafe function or block has been deprecated as it is now an unsafe function

    // test for fn() coercion where arguments match (no args)
    let depr_safe_fn_ptr: fn() = depr_safe; //~ WARN use of function `deprecated_safe::depr_safe` as a normal fn pointer has been deprecated as it is now an unsafe function

    // test that second usage still lints
    let depr_safe_fn_ptr: fn() = depr_safe; //~ WARN use of function `deprecated_safe::depr_safe` as a normal fn pointer has been deprecated as it is now an unsafe function

    // test for fn() coercion where arguments don't match, where an
    // intermediate fn() will be used
    let depr_safe_fn_ptr: fn() = depr_safe; //~ WARN use of function `deprecated_safe::depr_safe` as a normal fn pointer has been deprecated as it is now an unsafe function
    let depr_safe_fn_impl: Box<dyn Fn()> = Box::new(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    let mut depr_safe_fnmut_impl: Box<dyn FnMut()> = Box::new(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    let depr_safe_fnonce_impl: Box<dyn FnOnce()> = Box::new(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function

    // test that second usage still lints
    let depr_safe_fn_ptr: fn() = depr_safe; //~ WARN use of function `deprecated_safe::depr_safe` as a normal fn pointer has been deprecated as it is now an unsafe function
    let depr_safe_fn_impl: Box<dyn Fn()> = Box::new(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    let mut depr_safe_fnmut_impl: Box<dyn FnMut()> = Box::new(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    let depr_safe_fnonce_impl: Box<dyn FnOnce()> = Box::new(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function

    // these shouldn't lint, appropriate unsafe usage
    let unsafe_depr_safe_fn_ptr: unsafe fn() = depr_safe;
    fn_taking_unsafe_fn_ptr(depr_safe);
    unsafe {
        depr_safe();
    }

    // all of these coercions should lint
    fn_taking_fn_ptr(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a normal fn pointer has been deprecated as it is now an unsafe function
    fn_taking_dyn_fn_impl(Box::new(depr_safe)); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl(Box::new(depr_safe)); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl(Box::new(depr_safe)); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fn_impl(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function

    // ensure lint still fires if coerced again
    fn_taking_fn_ptr(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a normal fn pointer has been deprecated as it is now an unsafe function
    fn_taking_dyn_fn_impl(Box::new(depr_safe)); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl(Box::new(depr_safe)); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl(Box::new(depr_safe)); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fn_impl(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl(depr_safe); //~ WARN use of function `deprecated_safe::depr_safe` as a closure has been deprecated as it is now an unsafe function

    // test for dyn Fn() coercion where arguments don't match, where an
    // intermediate fn() will be used
    let fn_impl: Box<dyn Fn(OsString, OsString)> = Box::new(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnMut(OsString, OsString)> = Box::new(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnOnce(OsString, OsString)> = Box::new(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function

    // test that second usage still lints
    let fn_impl: Box<dyn Fn(OsString, OsString)> = Box::new(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnMut(OsString, OsString)> = Box::new(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    let fn_impl: Box<dyn FnOnce(OsString, OsString)> = Box::new(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function

    // test variant where coercion occurs on a function argument instead of a variable
    fn_taking_dyn_fn_impl_generic(Box::new(depr_safe_generic)); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl_generic(Box::new(depr_safe_generic)); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl_generic(Box::new(depr_safe_generic)); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function

    // test that second usage still lints
    fn_taking_dyn_fn_impl_generic(Box::new(depr_safe_generic)); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl_generic(Box::new(depr_safe_generic)); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl_generic(Box::new(depr_safe_generic)); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function

    // test for non-dyn Fn() coercion (no unsizing)
    fn_taking_fn_impl_generic(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl_generic(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl_generic(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function

    // test that second usage still lints
    fn_taking_fn_impl_generic(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl_generic(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl_generic(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function

    // usage without unsafe should lint
    depr_safe_generic("", ""); //~ WARN use of function `deprecated_safe::depr_safe_generic` without an unsafe function or block has been deprecated as it is now an unsafe function
    depr_safe_generic("", ""); //~ WARN use of function `deprecated_safe::depr_safe_generic` without an unsafe function or block has been deprecated as it is now an unsafe function

    // test for fn() coercion where arguments match (no args)
    let depr_safe_generic_fn_ptr: fn() = depr_safe; //~ WARN use of function `deprecated_safe::depr_safe` as a normal fn pointer has been deprecated as it is now an unsafe function

    // test that second usage still lints
    let depr_safe_generic_fn_ptr: fn() = depr_safe; //~ WARN use of function `deprecated_safe::depr_safe` as a normal fn pointer has been deprecated as it is now an unsafe function

    // test for fn() coercion where arguments don't match, where an
    // intermediate fn() will be used
    let depr_safe_generic_fn_ptr: fn(OsString, OsString) = depr_safe_generic; //~ WARN use of function `deprecated_safe::depr_safe_generic` as a normal fn pointer has been deprecated as it is now an unsafe function
    let depr_safe_generic_fn_impl: Box<dyn Fn(OsString, OsString)> = Box::new(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    let mut depr_safe_generic_fnmut_impl: Box<dyn FnMut(OsString, OsString)> =
        Box::new(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    let depr_safe_generic_fnonce_impl: Box<dyn FnOnce(OsString, OsString)> =
        Box::new(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function

    // test that second usage still lints
    let depr_safe_generic_fn_ptr: fn(OsString, OsString) = depr_safe_generic; //~ WARN use of function `deprecated_safe::depr_safe_generic` as a normal fn pointer has been deprecated as it is now an unsafe function
    let depr_safe_generic_fn_impl: Box<dyn Fn(OsString, OsString)> = Box::new(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    let mut depr_safe_generic_fnmut_impl: Box<dyn FnMut(OsString, OsString)> =
        Box::new(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    let depr_safe_generic_fnonce_impl: Box<dyn FnOnce(OsString, OsString)> =
        Box::new(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function

    // these shouldn't lint, appropriate unsafe usage
    let unsafe_depr_safe_generic_fn_ptr: unsafe fn(OsString, OsString) = depr_safe_generic;
    fn_taking_unsafe_fn_ptr_generic(depr_safe_generic);
    unsafe {
        depr_safe_generic("", "");
    }

    // all of these coercions should lint
    fn_taking_fn_ptr_generic(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a normal fn pointer has been deprecated as it is now an unsafe function
    fn_taking_dyn_fn_impl_generic(Box::new(depr_safe_generic)); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl_generic(Box::new(depr_safe_generic)); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl_generic(Box::new(depr_safe_generic)); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fn_impl_generic(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl_generic(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl_generic(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function

    // ensure lint still fires if coerced again
    fn_taking_fn_ptr_generic(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a normal fn pointer has been deprecated as it is now an unsafe function
    fn_taking_dyn_fn_impl_generic(Box::new(depr_safe_generic)); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnmut_impl_generic(Box::new(depr_safe_generic)); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_dyn_fnonce_impl_generic(Box::new(depr_safe_generic)); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fn_impl_generic(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnmut_impl_generic(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
    fn_taking_fnonce_impl_generic(depr_safe_generic); //~ WARN use of function `deprecated_safe::depr_safe_generic` as a closure has been deprecated as it is now an unsafe function
}

fn fn_taking_fn_ptr(_: fn()) {}
fn fn_taking_unsafe_fn_ptr(_: unsafe fn()) {}

fn fn_taking_dyn_fn_impl(_: Box<dyn Fn()>) {}
fn fn_taking_dyn_fnmut_impl(_: Box<dyn FnMut()>) {}
fn fn_taking_dyn_fnonce_impl(_: Box<dyn FnOnce()>) {}

fn fn_taking_fn_impl(_: impl Fn()) {}
fn fn_taking_fnmut_impl(_: impl FnMut()) {}
fn fn_taking_fnonce_impl(_: impl FnOnce()) {}

fn fn_taking_fn_ptr_generic(_: fn(OsString, OsString)) {}
fn fn_taking_unsafe_fn_ptr_generic(_: unsafe fn(OsString, OsString)) {}

fn fn_taking_dyn_fn_impl_generic(_: Box<dyn Fn(OsString, OsString)>) {}
fn fn_taking_dyn_fnmut_impl_generic(_: Box<dyn FnMut(OsString, OsString)>) {}
fn fn_taking_dyn_fnonce_impl_generic(_: Box<dyn FnOnce(OsString, OsString)>) {}

fn fn_taking_fn_impl_generic(_: impl Fn(OsString, OsString)) {}
fn fn_taking_fnmut_impl_generic(_: impl FnMut(OsString, OsString)) {}
fn fn_taking_fnonce_impl_generic(_: impl FnOnce(OsString, OsString)) {}
