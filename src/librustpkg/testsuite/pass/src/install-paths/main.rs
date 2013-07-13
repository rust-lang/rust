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
The test runner should check that, after `rustpkg install install-paths`
  with RUST_PATH undefined in the environment:
   * ./.rust/install_paths exists and is an executable
   * ./.rust/libinstall_paths exists and is a library
   * ./.rust/install_pathstest does not exist
   * ./.rust/install_pathsbench does not exist
   * install-paths/build/install_pathstest exists and is an executable
   * install-paths/build/install_pathsbench exists and is an executable
*/

use lib::f;

mod lib;

fn main() {
    f();
}
