// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the deprecated markers still have their old effect.

#![feature(rustc_attrs)]

use std::marker;

#[rustc_variance]
struct A<T>(marker::CovariantType<T>); //~ ERROR types=[[+];[];[]]

#[rustc_variance]
struct B<T>(marker::ContravariantType<T>); //~ ERROR types=[[-];[];[]]

#[rustc_variance]
struct C<T>(marker::InvariantType<T>); //~ ERROR types=[[o];[];[]]

#[rustc_variance]
struct D<'a>(marker::CovariantLifetime<'a>); //~ ERROR regions=[[+];[];[]]

#[rustc_variance]
struct E<'a>(marker::ContravariantLifetime<'a>); //~ ERROR regions=[[-];[];[]]

#[rustc_variance]
struct F<'a>(marker::InvariantLifetime<'a>); //~ ERROR regions=[[o];[];[]]

fn main() { }
