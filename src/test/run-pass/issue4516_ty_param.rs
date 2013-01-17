// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast - check-fast doesn't understand aux-build
// aux-build:issue4516_ty_param_lib.rs

// Trigger a bug concerning inlining of generic functions.
// The def-ids in type parameters were not being correctly
// resolved and hence when we checked the type of the closure
// variable (see the library mod) to determine if the value
// should be moved into the closure, trans failed to find
// the relevant kind bounds.

extern mod issue4516_ty_param_lib;
use issue4516_ty_param_lib::to_closure;
fn main() {
    to_closure(22)();
}
