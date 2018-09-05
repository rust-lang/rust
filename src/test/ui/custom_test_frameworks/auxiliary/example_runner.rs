// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Testable {
    fn name(&self) -> String;
    fn run(&self) -> Option<String>; // None will be success, Some is the error message
}

pub fn runner(tests: &[&dyn Testable]) {
    for t in tests {
        print!("{}........{}", t.name(), t.run().unwrap_or_else(|| "SUCCESS".to_string()));
    }
}
