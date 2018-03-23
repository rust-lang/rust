// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(no_core, lang_items)]
#![crate_type="rlib"]
#![no_core]

pub static STATIC_BOOL: bool = true;

pub static mut STATIC_MUT_BOOL: bool = true;

const CONST_BOOL: bool = true;
pub static CONST_BOOL_REF: &'static bool = &CONST_BOOL;


#[lang = "sized"]
trait Sized {}

#[lang = "copy"]
trait Copy {}

#[lang = "freeze"]
trait Freeze {}

#[lang = "sync"]
trait Sync {}
impl Sync for bool {}
impl Sync for &'static bool {}

#[lang="drop_in_place"]
pub unsafe fn drop_in_place<T: ?Sized>(_: *mut T) { }
