// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Concurrency-enabled mechanisms and primitives.
 */

#[allow(missing_doc)];
#[feature(globs)];

#[crate_id = "concurrency#0.10-pre"];
#[crate_type = "rlib"];
#[crate_type = "lib"];
#[license = "MIT/ASL2"];

pub mod arc;
pub mod sync;
pub mod comm;
pub mod task_pool;
pub mod future;
