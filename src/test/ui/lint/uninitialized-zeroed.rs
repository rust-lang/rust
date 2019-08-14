// ignore-tidy-linelength
// This test checks that calling `mem::{uninitialized,zeroed}` with certain types results
// in a lint.

#![feature(never_type)]
#![allow(deprecated)]
#![deny(invalid_value)]

use std::mem::{self, MaybeUninit};

enum Void {}

struct Ref(&'static i32);
struct RefPair((&'static i32, i32));

struct Wrap<T> { wrapped: T }
enum WrapEnum<T> { Wrapped(T) }

#[allow(unused)]
fn generic<T: 'static>() {
    unsafe {
        let _val: &'static T = mem::zeroed(); //~ ERROR: does not permit zero-initialization
        let _val: &'static T = mem::uninitialized(); //~ ERROR: does not permit being left uninitialized

        let _val: Wrap<&'static T> = mem::zeroed(); //~ ERROR: does not permit zero-initialization
        let _val: Wrap<&'static T> = mem::uninitialized(); //~ ERROR: does not permit being left uninitialized
    }
}

fn main() {
    unsafe {
        let _val: ! = mem::zeroed(); //~ ERROR: does not permit zero-initialization
        let _val: ! = mem::uninitialized(); //~ ERROR: does not permit being left uninitialized

        let _val: (i32, !) = mem::zeroed(); //~ ERROR: does not permit zero-initialization
        let _val: (i32, !) = mem::uninitialized(); //~ ERROR: does not permit being left uninitialized

        let _val: Void = mem::zeroed(); //~ ERROR: does not permit zero-initialization
        let _val: Void = mem::uninitialized(); //~ ERROR: does not permit being left uninitialized

        let _val: &'static i32 = mem::zeroed(); //~ ERROR: does not permit zero-initialization
        let _val: &'static i32 = mem::uninitialized(); //~ ERROR: does not permit being left uninitialized

        let _val: Ref = mem::zeroed(); //~ ERROR: does not permit zero-initialization
        let _val: Ref = mem::uninitialized(); //~ ERROR: does not permit being left uninitialized

        let _val: fn() = mem::zeroed(); //~ ERROR: does not permit zero-initialization
        let _val: fn() = mem::uninitialized(); //~ ERROR: does not permit being left uninitialized

        let _val: Wrap<fn()> = mem::zeroed(); //~ ERROR: does not permit zero-initialization
        let _val: Wrap<fn()> = mem::uninitialized(); //~ ERROR: does not permit being left uninitialized

        let _val: WrapEnum<fn()> = mem::zeroed(); //~ ERROR: does not permit zero-initialization
        let _val: WrapEnum<fn()> = mem::uninitialized(); //~ ERROR: does not permit being left uninitialized

        let _val: Wrap<(RefPair, i32)> = mem::zeroed(); //~ ERROR: does not permit zero-initialization
        let _val: Wrap<(RefPair, i32)> = mem::uninitialized(); //~ ERROR: does not permit being left uninitialized

        // Some types that should work just fine.
        let _val: Option<&'static i32> = mem::zeroed();
        let _val: Option<fn()> = mem::zeroed();
        let _val: MaybeUninit<&'static i32> = mem::zeroed();
        let _val: bool = mem::zeroed();
        let _val: i32 = mem::zeroed();
    }
}
