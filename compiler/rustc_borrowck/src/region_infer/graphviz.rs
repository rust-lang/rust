//! This module provides linkage between RegionInferenceContext and
//! `rustc_graphviz` traits, specialized to attaching borrowck analysis
//! data to rendered labels.

use std::borrow::Cow;
use std::io::{self, Write};

use super::*;
use crate::constraints::OutlivesConstraint;
use rustc_graphviz as dot;

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Write out the region constraint graph.
    crate fn dump_graphviz_raw_constraints(&self, mut w: &mut dyn Write) -> io::Result<()> {
        dot::render(&RawConstraints { regioncx: self }, &mut w)
    }

    /// Write out the region constraint graph.
    crate fn dump_graphviz_scc_constraints(&self, mut w: &mut dyn Write) -> io::Result<()> {
        let mut nodes_per_scc: IndexVec<ConstraintSccIndex, _> =
            self.constraint_sccs.all_sccs().map(|_| Vec::new()).collect();

        for region in self.definitions.indices() {
            let scc = self.constraint_sccs.scc(region);
            nodes_per_scc[scc].push(region);
        }

        dot::render(&SccConstraints { regioncx: self, nodes_per_scc }, &mut w)
    }
}

struct RawConstraints<'a, 'tcx> {
    regioncx: &'a RegionInferenceContext<'tcx>,
}

impl<'a, 'this, 'tcx> dot::Labeller<'this> for RawConstraints<'a, 'tcx> {
    type Node = RegionVid;
    type Edge = OutlivesConstraint<'tcx>;

    fn graph_id(&'this self) -> dot::Id<'this> {
        dot::Id::new("RegionInferenceContext").unwrap()
    }
    fn node_id(&'this self, n: &RegionVid) -> dot::Id<'this> {
        dot::Id::new(format!("r{}", n.index())).unwrap()
    }
    fn node_shape(&'this self, _node: &RegionVid) -> Option<dot::LabelText<'this>> {
        Some(dot::LabelText::LabelStr(Cow::Borrowed("box")))
    }
    fn node_label(&'this self, n: &RegionVid) -> dot::LabelText<'this> {
        dot::LabelText::LabelStr(format!("{:?}", n).into())
    }
    fn edge_label(&'this self, e: &OutlivesConstraint<'tcx>) -> dot::LabelText<'this> {
        dot::LabelText::LabelStr(format!("{:?}", e.locations).into())
    }
}

impl<'a, 'this, 'tcx> dot::GraphWalk<'this> for RawConstraints<'a, 'tcx> {
    type Node = RegionVid;
    type Edge = OutlivesConstraint<'tcx>;

    fn nodes(&'this self) -> dot::Nodes<'this, RegionVid> {
        let vids: Vec<RegionVid> = self.regioncx.definitions.indices().collect();
        vids.into()
    }
    fn edges(&'this self) -> dot::Edges<'this, OutlivesConstraint<'tcx>> {
        (&self.regioncx.constraints.outlives().raw[..]).into()
    }

    // Render `a: b` as `a -> b`, indicating the flow
    // of data during inference.

    fn source(&'this self, edge: &OutlivesConstraint<'tcx>) -> RegionVid {
        edge.sup
    }

    fn target(&'this self, edge: &OutlivesConstraint<'tcx>) -> RegionVid {
        edge.sub
    }
}

struct SccConstraints<'a, 'tcx> {
    regioncx: &'a RegionInferenceContext<'tcx>,
    nodes_per_scc: IndexVec<ConstraintSccIndex, Vec<RegionVid>>,
}

impl<'a, 'this, 'tcx> dot::Labeller<'this> for SccConstraints<'a, 'tcx> {
    type Node = ConstraintSccIndex;
    type Edge = (ConstraintSccIndex, ConstraintSccIndex);

    fn graph_id(&'this self) -> dot::Id<'this> {
        dot::Id::new("RegionInferenceContext".to_string()).unwrap()
    }
    fn node_id(&'this self, n: &ConstraintSccIndex) -> dot::Id<'this> {
        dot::Id::new(format!("r{}", n.index())).unwrap()
    }
    fn node_shape(&'this self, _node: &ConstraintSccIndex) -> Option<dot::LabelText<'this>> {
        Some(dot::LabelText::LabelStr(Cow::Borrowed("box")))
    }
    fn node_label(&'this self, n: &ConstraintSccIndex) -> dot::LabelText<'this> {
        let nodes = &self.nodes_per_scc[*n];
        dot::LabelText::LabelStr(format!("{:?} = {:?}", n, nodes).into())
    }
}

impl<'a, 'this, 'tcx> dot::GraphWalk<'this> for SccConstraints<'a, 'tcx> {
    type Node = ConstraintSccIndex;
    type Edge = (ConstraintSccIndex, ConstraintSccIndex);

    fn nodes(&'this self) -> dot::Nodes<'this, ConstraintSccIndex> {
        let vids: Vec<ConstraintSccIndex> = self.regioncx.constraint_sccs.all_sccs().collect();
        vids.into()
    }
    fn edges(&'this self) -> dot::Edges<'this, (ConstraintSccIndex, ConstraintSccIndex)> {
        let edges: Vec<_> = self
            .regioncx
            .constraint_sccs
            .all_sccs()
            .flat_map(|scc_a| {
                self.regioncx
                    .constraint_sccs
                    .successors(scc_a)
                    .iter()
                    .map(move |&scc_b| (scc_a, scc_b))
            })
            .collect();

        edges.into()
    }

    // Render `a: b` as `a -> b`, indicating the flow
    // of data during inference.

    fn source(&'this self, edge: &(ConstraintSccIndex, ConstraintSccIndex)) -> ConstraintSccIndex {
        edge.0
    }

    fn target(&'this self, edge: &(ConstraintSccIndex, ConstraintSccIndex)) -> ConstraintSccIndex {
        edge.1
    }
}
