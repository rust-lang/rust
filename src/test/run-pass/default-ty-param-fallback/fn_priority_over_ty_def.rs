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

use std::path::Path;

enum Opt<T=String> {
    Som(T),
    Non,
}

fn main() {
    // Defaults on the type definiton work, as long no other params are interfering.
    let _ = Opt::Non;
    let _: Opt<_> = Opt::Non;

    func1(None);
    func2(Opt::Non);
}

// Defaults on fns take precedence.
fn func1<P: AsRef<Path> = String>(p: Option<P>) { }
fn func2<P: AsRef<Path> = String>(p: Opt<P>) { }
