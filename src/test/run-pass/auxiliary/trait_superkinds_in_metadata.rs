// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test library crate for cross-crate usages of traits inheriting
// from the builtin kinds. Mostly tests metadata correctness.

#![crate_type="lib"]

pub trait RequiresShare : Sync { }
pub trait RequiresRequiresShareAndSend : RequiresShare + Send { }
pub trait RequiresCopy : Copy { }
