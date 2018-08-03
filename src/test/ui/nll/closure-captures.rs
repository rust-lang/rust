// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Some cases with closures that might be problems

#![allow(unused)]
#![feature(nll)]

// Should have one error per assignment

fn one_closure(x: i32) {
    ||
    x = 1; //~ ERROR
    move ||
    x = 1; //~ ERROR
}

fn two_closures(x: i32) {
    || {
        ||
        x = 1; //~ ERROR
    };
    move || {
        ||
        x = 1; //~ ERROR
    };
}

fn fn_ref<F: Fn()>(f: F) -> F { f }

fn two_closures_ref_mut(mut x: i32) {
    fn_ref(|| {
        || //~ ERROR
         x = 1;}
    );
    fn_ref(move || {
        ||  //~ ERROR
    x = 1;});
}

// This still gives two messages, but it requires two things to be fixed.
fn two_closures_ref(x: i32) {
    fn_ref(|| {
        || //~ ERROR
         x = 1;} //~ ERROR
    );
    fn_ref(move || {
        ||  //~ ERROR
    x = 1;}); //~ ERROR
}

fn two_closures_two_refs(x: &mut i32) {
    fn_ref(|| {
        || //~ ERROR
        *x = 1;});
    fn_ref(move || {
        || //~ ERROR
        *x = 1;});
}

fn main() {}
