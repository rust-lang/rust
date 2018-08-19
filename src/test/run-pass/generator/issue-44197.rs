// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generators, generator_trait)]

use std::ops::{ Generator, GeneratorState };

fn foo(_: &str) -> String {
    String::new()
}

fn bar(baz: String) -> impl Generator<Yield = String, Return = ()> {
    move || {
        yield foo(&baz);
    }
}

fn foo2(_: &str) -> Result<String, ()> {
    Err(())
}

fn bar2(baz: String) -> impl Generator<Yield = String, Return = ()> {
    move || {
        if let Ok(quux) = foo2(&baz) {
            yield quux;
        }
    }
}

fn main() {
    unsafe {
        assert_eq!(bar(String::new()).resume(), GeneratorState::Yielded(String::new()));
        assert_eq!(bar2(String::new()).resume(), GeneratorState::Complete(()));
    }
}
