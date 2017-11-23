// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(default_type_parameter_fallback)]

struct Foo;

impl Foo {
    fn method<A:Default=String>(&self) -> A {
        A::default()
    }
}

fn main() {
    // Bug: Replacing with Foo.method::<_>() fails.
    // Suspicion: astconv and tycheck are considering this a "A provided type parameter.",
    // resulting in type_var_for_def not being called and a TypeInference var being created
    // rather than a TypeParameterDefinition var.
    let _ = Foo.method();
}
