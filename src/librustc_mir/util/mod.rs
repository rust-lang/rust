// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod borrowck_errors;
pub mod elaborate_drops;
pub mod def_use;
pub mod patch;

mod alignment;
mod graphviz;
pub(crate) mod pretty;
pub mod liveness;
pub mod collect_writes;

pub use self::alignment::is_disaligned;
pub use self::pretty::{dump_enabled, dump_mir, write_mir_pretty, PassWhere};
pub use self::graphviz::{write_mir_graphviz};
pub use self::graphviz::write_node_label as write_graphviz_node_label;
