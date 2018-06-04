// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module provides linkage between RegionInferenceContext and
//! libgraphviz traits, specialized to attaching borrowck analysis
//! data to rendered labels.

use dot::{self, IntoCow};
use rustc_data_structures::indexed_vec::Idx;
use std::borrow::Cow;
use std::io::{self, Write};
use super::*;

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Write out the region constraint graph.
    pub(crate) fn dump_graphviz(&self, mut w: &mut dyn Write) -> io::Result<()> {
        dot::render(self, &mut w)
    }
}

impl<'this, 'tcx> dot::Labeller<'this> for RegionInferenceContext<'tcx> {
    type Node = RegionVid;
    type Edge = OutlivesConstraint;

    fn graph_id(&'this self) -> dot::Id<'this> {
        dot::Id::new(format!("RegionInferenceContext")).unwrap()
    }
    fn node_id(&'this self, n: &RegionVid) -> dot::Id<'this> {
        dot::Id::new(format!("r{}", n.index())).unwrap()
    }
    fn node_shape(&'this self, _node: &RegionVid) -> Option<dot::LabelText<'this>> {
        Some(dot::LabelText::LabelStr(Cow::Borrowed("box")))
    }
    fn node_label(&'this self, n: &RegionVid) -> dot::LabelText<'this> {
        dot::LabelText::LabelStr(format!("{:?}", n).into_cow())
    }
    fn edge_label(&'this self, e: &OutlivesConstraint) -> dot::LabelText<'this> {
        dot::LabelText::LabelStr(format!("{:?}", e.point).into_cow())
    }
}

impl<'this, 'tcx> dot::GraphWalk<'this> for RegionInferenceContext<'tcx> {
    type Node = RegionVid;
    type Edge = OutlivesConstraint;

    fn nodes(&'this self) -> dot::Nodes<'this, RegionVid> {
        let vids: Vec<RegionVid> = self.definitions.indices().collect();
        vids.into_cow()
    }
    fn edges(&'this self) -> dot::Edges<'this, OutlivesConstraint> {
        (&self.constraints.raw[..]).into_cow()
    }

    // Render `a: b` as `a <- b`, indicating the flow
    // of data during inference.

    fn source(&'this self, edge: &OutlivesConstraint) -> RegionVid {
        edge.sub
    }

    fn target(&'this self, edge: &OutlivesConstraint) -> RegionVid {
        edge.sup
    }
}
