// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dep_graph::{DepGraph, DepNode, DepTrackingMap, DepTrackingMapConfig};
use hir::def_id::DefId;
use mir::repr::Mir;
use std::marker::PhantomData;

pub struct MirMap<'tcx> {
    pub map: DepTrackingMap<MirMapConfig<'tcx>>,
}

impl<'tcx> MirMap<'tcx> {
    pub fn new(graph: DepGraph) -> Self {
        MirMap {
            map: DepTrackingMap::new(graph)
        }
    }
}

pub struct MirMapConfig<'tcx> {
    data: PhantomData<&'tcx ()>
}

impl<'tcx> DepTrackingMapConfig for MirMapConfig<'tcx> {
    type Key = DefId;
    type Value = Mir<'tcx>;
    fn to_dep_node(key: &DefId) -> DepNode<DefId> {
        DepNode::Mir(*key)
    }
}
