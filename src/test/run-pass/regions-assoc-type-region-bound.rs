// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the compiler considers the 'a bound declared in the
// trait. Issue #20890.

trait Foo<'a> {
    type Value: 'a;

    fn get(&self) -> &'a Self::Value;
}

fn takes_foo<'a,F: Foo<'a>>(f: &'a F) {
    // This call would be illegal, because it results in &'a F::Value,
    // and the only way we know that `F::Value : 'a` is because of the
    // trait declaration.

    f.get();
}

fn main() { }
