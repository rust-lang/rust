// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod debug;
mod dep_node;
mod dep_tracking_map;
mod edges;
mod graph;
mod query;
mod raii;
mod shadow;
mod thread;
mod visit;

pub use self::dep_tracking_map::{DepTrackingMap, DepTrackingMapConfig};
pub use self::dep_node::DepNode;
pub use self::dep_node::WorkProductId;
pub use self::graph::DepGraph;
pub use self::graph::WorkProduct;
pub use self::query::DepGraphQuery;
pub use self::visit::visit_all_item_likes_in_krate;
pub use self::raii::DepTask;
