// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![allow(warnings)]

type Different<'a, 'b> = &'a mut (&'a (), &'b ());
type Same<'a> = Different<'a, 'a>;

fn with_closure_expecting_different<F>(_: F)
    where F: for<'a, 'b> FnOnce(Different<'a, 'b>)
{
}

fn with_closure_expecting_different_anon<F>(_: F)
    where F: FnOnce(Different<'_, '_>)
{
}

fn supplying_nothing_expecting_anon() {
    with_closure_expecting_different_anon(|x: Different| {
    })
}

fn supplying_nothing_expecting_named() {
    with_closure_expecting_different(|x: Different| {
    })
}

fn supplying_underscore_expecting_anon() {
    with_closure_expecting_different_anon(|x: Different<'_, '_>| {
    })
}

fn supplying_underscore_expecting_named() {
    with_closure_expecting_different(|x: Different<'_, '_>| {
    })
}

fn main() { }
