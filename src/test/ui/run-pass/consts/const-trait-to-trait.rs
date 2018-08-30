// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #24644 - block causes a &Trait -> &Trait coercion:
trait Trait {}

struct Bar;
impl Trait for Bar {}

fn main() {
    let x: &[&Trait] = &[{ &Bar }];
}

// Issue #25748 - the cast causes an &Encoding -> &Encoding coercion:
pub struct UTF8Encoding;
pub const UTF_8: &'static UTF8Encoding = &UTF8Encoding;
pub trait Encoding {}
impl Encoding for UTF8Encoding {}
pub fn f() -> &'static Encoding { UTF_8 as &'static Encoding }

// Root of the problem: &Trait -> &Trait coercions:
const FOO: &'static Trait = &Bar;
const BAR: &'static Trait = FOO;
fn foo() { let _x = BAR; }
