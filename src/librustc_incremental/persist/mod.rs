// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! When in incremental mode, this pass dumps out the dependency graph
//! into the given directory. At the same time, it also hashes the
//! various HIR nodes.

mod data;
mod directory;
mod dirty_clean;
mod load;
mod save;
mod util;

pub use self::load::load_dep_graph;
pub use self::save::save_dep_graph;
