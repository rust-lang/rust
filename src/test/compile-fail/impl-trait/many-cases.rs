// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A simple test for testing many permutations of allowedness of
//! impl Trait
#![feature(conservative_impl_trait, universal_impl_trait, dyn_trait)]
use std::fmt::Debug;

// Allowed
fn simple_universal(_: impl Debug) { panic!() }

// Allowed
fn simple_existential() -> impl Debug { panic!() }

// Allowed
fn collection_universal(_: Vec<impl Debug>) { panic!() }

// Allowed
fn collection_existential() -> Vec<impl Debug> { panic!() }

// Disallowed
fn fn_type_universal(_: fn(impl Debug)) { panic!() }
//~^ ERROR `impl Trait` not allowed outside of function and inherent method return types

// Disallowed
fn fn_type_existential() -> fn(impl Debug) { panic!() }
//~^ ERROR `impl Trait` not allowed outside of function and inherent method return types

// Allowed
fn dyn_universal(_: &dyn Iterator<Item = impl Debug>) { panic!() }

// Disallowed
fn dyn_fn_trait(_: &dyn Fn(impl Debug)) { panic!() }
//~^ ERROR `impl Trait` not allowed outside of function and inherent method return types

// Allowed
fn nested_universal(_: impl Iterator<Item = impl Iterator>) { panic!() }

// Allowed
fn nested_existential() -> impl IntoIterator<Item = impl IntoIterator> {
    vec![vec![0; 10], vec![12; 7], vec![8; 3]]
}

// Disallowed
fn universal_fn_trait(_: impl Fn(impl Debug)) { panic!() }
//~^ ERROR `impl Trait` not allowed outside of function and inherent method return types

// Disallowed
struct ImplMember { x: impl Debug }
//~^ ERROR `impl Trait` not allowed outside of function and inherent method return types

// Disallowed
trait Universal {
    // FIXME, should error?
    fn universal(impl Debug);
}

// Disallowed
trait Existential {
    fn existential() -> impl Debug;
    //~^ ERROR `impl Trait` not allowed outside of function and inherent method return types
}

fn main() {}
