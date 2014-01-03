// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern "C" void rust_cxx_throw() {
  throw 0;
}

typedef void *(rust_try_fn)(void*, void*);

extern "C" void
rust_cxx_try(rust_try_fn f, void *fptr, void *env) {
  try {
    f(fptr, env);
  } catch (int t) {
  }
}
