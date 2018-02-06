// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that default and negative trait implementations are gated by
// `optin_builtin_traits` feature gate

struct DummyStruct;

trait DummyTrait {
    fn dummy(&self) {}
}

auto trait AutoDummyTrait {}
//~^ ERROR auto traits are experimental and possibly buggy

impl !DummyTrait for DummyStruct {}
//~^ ERROR negative trait bounds are not yet fully implemented; use marker types for now

fn main() {}
