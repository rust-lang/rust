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

auto trait Auto {}
unsafe auto trait AutoUnsafe {}

impl !Auto for bool {}
impl !AutoUnsafe for bool {}

struct AutoBool(bool);

impl Auto for AutoBool {}
unsafe impl AutoUnsafe for AutoBool {}

fn take_auto<T: Auto>(_: T) {}
fn take_auto_unsafe<T: AutoUnsafe>(_: T) {}

fn main() {
    // Parse inside functions.
    auto trait AutoInner {}
    unsafe auto trait AutoUnsafeInner {}

    take_auto(0);
    take_auto(AutoBool(true));
    take_auto_unsafe(0);
    take_auto_unsafe(AutoBool(true));

    /// Auto traits are allowed in trait object bounds.
    let _: &(Send + Auto) = &0;
}
