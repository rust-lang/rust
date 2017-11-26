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

use std::path::Path;
use std::mem::size_of;

enum Opt<T=String> {
    Som(T),
    Non,
}

fn main() {
    // func1 and func2 cannot agree so we fail.
    // NB: While it would be future-proof wrt API evolution to use the `String` default in `Opt`,
    // that is not future proof wrt to compiler evolution,
    // such as making priority levels more fine-grained or just fixing bugs.
    let x = Opt::Non;
    func1(&x);
    func2(&x);
}

fn func1<P: AsRef<Path> = &'static str>(_: &Opt<P>) { }

fn func2<P: AsRef<Path> = &'static &'static str>(_: &Opt<P>) { }
