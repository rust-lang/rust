//! This module provides linkage between RegionInferenceContext and
//! `rustc_graphviz` traits, specialized to attaching borrowck analysis
//! data to rendered labels.

use std::borrow::Cow;
use std::io::{self, Write};

use itertools::Itertools;
use rustc_graphviz as dot;
use rustc_middle::ty::UniverseIndex;

use super::*;

fn render_outlives_constraint(constraint: &OutlivesConstraint<'_>) -> String {
    if let ConstraintCategory::OutlivesUnnameablePlaceholder(unnameable) = constraint.category {
        format!("{unnameable:?} unnameable")
    } else {
        match constraint.locations {
            Locations::All(_) => "All(...)".to_string(),
            Locations::Single(loc) => format!("{loc:?}"),
        }
    }
}

fn render_universe(u: UniverseIndex) -> String {
    if u.is_root() {
        return "".to_string();
    }

    format!("/{:?}", u)
}

fn render_region_vid<'tcx>(
    tcx: TyCtxt<'tcx>,
    rvid: RegionVid,
    regioncx: &RegionInferenceContext<'tcx>,
) -> String {
    let universe_str = render_universe(regioncx.region_definition(rvid).universe);

    let external_name_str = if let Some(external_name) =
        regioncx.region_definition(rvid).external_name.and_then(|e| e.get_name(tcx))
    {
        format!(" ({external_name})")
    } else {
        "".to_string()
    };

    let extra_info = match regioncx.region_definition(rvid).origin {
        NllRegionVariableOrigin::FreeRegion => "".to_string(),
        NllRegionVariableOrigin::Placeholder(p) => match p.bound.kind {
            ty::BoundRegionKind::Named(def_id) => {
                format!(" (for<{}>)", tcx.item_name(def_id))
            }
            ty::BoundRegionKind::ClosureEnv | ty::BoundRegionKind::Anon => " (for<'_>)".to_string(),
            ty::BoundRegionKind::NamedAnon(_) => {
                bug!("only used for pretty printing")
            }
        },
        NllRegionVariableOrigin::Existential { name: Some(name), .. } => format!(" (ex<{name}>)"),
        NllRegionVariableOrigin::Existential { .. } => format!(" (ex<'_>)"),
    };

    format!("{:?}{universe_str}{external_name_str}{extra_info}", rvid)
}

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Write out the region constraint graph.
    pub(crate) fn dump_graphviz_raw_constraints(
        &self,
        tcx: TyCtxt<'tcx>,
        mut w: &mut dyn Write,
    ) -> io::Result<()> {
        dot::render(&RawConstraints { tcx, regioncx: self }, &mut w)
    }

    /// Write out the region constraint SCC graph.
    pub(crate) fn dump_graphviz_scc_constraints(
        &self,
        tcx: TyCtxt<'tcx>,
        mut w: &mut dyn Write,
    ) -> io::Result<()> {
        let mut nodes_per_scc: IndexVec<ConstraintSccIndex, _> =
            self.constraint_sccs.all_sccs().map(|_| Vec::new()).collect();

        for region in self.definitions.indices() {
            let scc = self.constraint_sccs.scc(region);
            nodes_per_scc[scc].push(region);
        }

        dot::render(&SccConstraints { tcx, regioncx: self, nodes_per_scc }, &mut w)
    }
}

struct RawConstraints<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
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
        dot::LabelText::LabelStr(render_region_vid(self.tcx, *n, self.regioncx).into())
    }
    fn edge_label(&'this self, e: &OutlivesConstraint<'tcx>) -> dot::LabelText<'this> {
        dot::LabelText::LabelStr(render_outlives_constraint(e).into())
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
    tcx: TyCtxt<'tcx>,
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
        let nodes_str = self.nodes_per_scc[*n]
            .iter()
            .map(|n| render_region_vid(self.tcx, *n, self.regioncx))
            .join(", ");
        dot::LabelText::LabelStr(format!("SCC({n}) = {{{nodes_str}}}", n = n.as_usize()).into())
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
