// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(optin_builtin_traits)]
#![feature(specialization)]

struct S;
struct Z;

default impl S {} //~ ERROR inherent impls cannot be default

default unsafe impl Send for S {} //~ ERROR impls of auto traits cannot be default
default impl !Send for Z {} //~ ERROR impls of auto traits cannot be default

trait Tr {}
default impl !Tr for S {} //~ ERROR negative impls are only allowed for auto traits
