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

fn func<P: AsRef<Path> = String>(p: Option<P>) {
    match p {
        None => { println!("None"); }
        Some(path) => { println!("{:?}", path.as_ref()); }
    }
}

fn main() {
    // Dont fallback to future-proof against default on `noner`.
    func(noner());
}

fn noner<T>() -> Option<T> { None }
