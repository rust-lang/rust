// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(optin_builtin_traits)]

auto trait Generic<T> {}
//~^ ERROR auto traits cannot have generics
//~^^ traits with auto impls (`e.g. impl Trait for ..`) can not have type parameters
auto trait Bound : Copy {}
//~^ ERROR auto traits cannot have super traits
//~^^ traits with auto impls (`e.g. impl Trait for ..`) cannot have predicates
auto trait MyTrait { fn foo() {} }
//~^ ERROR auto traits cannot contain items
//~^^ traits with default impls (`e.g. impl Trait for ..`) must have no methods or associated items
fn main() {}
