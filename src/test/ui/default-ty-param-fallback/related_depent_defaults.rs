// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
// compile-flags: --error-format=human

#![feature(default_type_parameter_fallback)]

#[derive(Copy, Clone)]
enum Opt<T=usize> {
    Som(T),
    Non,
}

fn main() {
    let a = Opt::Non;
    let b = Opt::Non;
    func1(a, b);
    func2(a, b);

    let c = Opt::Non;
    let d = Opt::Non;
    func1(c, d);
    func2(d, c);
}

fn func1<X = u32, Y = X>(_: Opt<X>, _: Opt<Y>) {
}

fn func2<X = usize, Y = X>(_: Opt<X>, _: Opt<Y>) {
}

