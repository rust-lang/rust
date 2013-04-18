// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
The test runner should check that, after `rustpkg build fancy-lib`:
  * testsuite/fancy-lib/build/ exists
  * testsuite/fancy-lib/build/ contains a file called generated.rs
  * testsuite/fancy-lib/build/ contains a library named libfancy_lib
  * testsuite/fancy-lib/build/ does not contain an executable
  *
*/

extern mod std;

pub mod foo;
pub mod bar;
#[path = "build/generated.rs"] pub mod generated;
