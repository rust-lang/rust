// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// min-llvm-version 3.9

use std::ops::Deref;

fn main() {
    if env_var("FOOBAR").as_ref().map(Deref::deref).ok() == Some("yes") {
        panic!()
    }

    let env_home: Result<String, ()> = Ok("foo-bar-baz".to_string());
    let env_home = env_home.as_ref().map(Deref::deref).ok();

    if env_home == Some("") { panic!() }
}

#[inline(never)]
fn env_var(s: &str) -> Result<String, VarError> {
    Err(VarError::NotPresent)
}

pub enum VarError {
    NotPresent,
    NotUnicode(String),
}
