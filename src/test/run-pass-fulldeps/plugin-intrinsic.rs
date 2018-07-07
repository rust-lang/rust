// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:plugin_intrinsic_codegen.rs
// ignore-stage1

#![feature(plugin, intrinsics)]
#![plugin(plugin_intrinsic_codegen)]

extern "rust-intrinsic" {
  /// Returns the secret value.
  fn get_secret_value() -> u64;
}

fn main() {
  const SECRET_VALUE: u64 = 4;
  assert_eq!(unsafe { get_secret_value() }, SECRET_VALUE);
}
