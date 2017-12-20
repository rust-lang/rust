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

struct Vac<T>(Vec<T>);

impl<T=usize> Vac<T> {
    fn new() -> Self {
        Vac(Vec::new())
    }
}

trait Foo { }
trait Bar { }

impl<T:Bar=usize> Foo for Vac<T> {}
impl Bar for usize {}

fn takes_foo<F:Foo>(_: F) {}

fn main() {
    let x = Vac::new(); // x: Vac<$0>
    // adds oblig Vac<$0> : Foo,
    // and applies the default of `impl<T> Vac<T>` and
    // `impl<T:Bar> Foo for Vac<T>`, which must agree.
    //
    // The default of F in takes_foo<F> makes no difference here,
    // because the corresponding inference var will be generalized
    // to Vac<_>.
    takes_foo(x);
}
