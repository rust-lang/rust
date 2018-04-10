// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//revisions: ast mir
//[mir] compile-flags: -Z borrowck=mir

#![feature(rustc_attrs)]

// Here is arielb1's basic example from rust-lang/rust#27282
// that AST borrowck is flummoxed by:

fn should_reject_destructive_mutate_in_guard() {
    match Some(&4) {
        None => {},
        ref mut foo if {
            (|| { let bar = foo; bar.take() })();
            //[mir]~^ ERROR cannot move out of borrowed content [E0507]
            false } => { },
        Some(s) => std::process::exit(*s),
    }
}

// Here below is a case that needs to keep working: we only use the
// binding via immutable-borrow in the guard, and we mutate in the arm
// body.
fn allow_mutate_in_arm_body() {
    match Some(&4) {
        None => {},
        ref mut foo if foo.is_some() && false => { foo.take(); () }
        Some(s) => std::process::exit(*s),
    }
}

// Here below is a case that needs to keep working: we only use the
// binding via immutable-borrow in the guard, and we move into the arm
// body.
fn allow_move_into_arm_body() {
    match Some(&4) {
        None => {},
        mut foo if foo.is_some() && false => { foo.take(); () }
        Some(s) => std::process::exit(*s),
    }
}

// Since this is a compile-fail test that is explicitly encoding the
// different behavior of AST- vs MIR-borrowck where AST-borrowck does
// not error, we need to use rustc_error to placate the test harness
// that wants *some* error to occur.
#[rustc_error]
fn main() { //[ast]~ ERROR compilation successful
    should_reject_destructive_mutate_in_guard();
    allow_mutate_in_arm_body();
    allow_move_into_arm_body();
}
