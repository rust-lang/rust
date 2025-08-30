use std::io;

use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_index::IndexVec;
use rustc_middle::mir::pretty::{MirDumper, PassWhere, PrettyPrintMirOptions};
use rustc_middle::mir::{Body, Location};
use rustc_middle::ty::{RegionVid, TyCtxt};
use rustc_mir_dataflow::points::PointIndex;
use rustc_session::config::MirIncludeSpans;

use crate::borrow_set::BorrowSet;
use crate::constraints::OutlivesConstraint;
use crate::polonius::{
    LocalizedOutlivesConstraint, LocalizedOutlivesConstraintSet, PoloniusDiagnosticsContext,
};
use crate::region_infer::values::LivenessValues;
use crate::type_check::Locations;
use crate::{BorrowckInferCtxt, ClosureRegionRequirements, RegionInferenceContext};

/// `-Zdump-mir=polonius` dumps MIR annotated with NLL and polonius specific information.
pub(crate) fn dump_polonius_mir<'tcx>(
    infcx: &BorrowckInferCtxt<'tcx>,
    body: &Body<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    closure_region_requirements: &Option<ClosureRegionRequirements<'tcx>>,
    borrow_set: &BorrowSet<'tcx>,
    polonius_diagnostics: Option<&PoloniusDiagnosticsContext>,
) {
    let tcx = infcx.tcx;
    if !tcx.sess.opts.unstable_opts.polonius.is_next_enabled() {
        return;
    }

    let Some(dumper) = MirDumper::new(tcx, "polonius", body) else { return };

    let polonius_diagnostics =
        polonius_diagnostics.expect("missing diagnostics context with `-Zpolonius=next`");

    let extra_data = &|pass_where, out: &mut dyn io::Write| {
        emit_polonius_mir(
            tcx,
            regioncx,
            closure_region_requirements,
            borrow_set,
            &polonius_diagnostics.localized_outlives_constraints,
            pass_where,
            out,
        )
    };
    // We want the NLL extra comments printed by default in NLL MIR dumps. Specifying `-Z
    // mir-include-spans` on the CLI still has priority.
    let options = PrettyPrintMirOptions {
        include_extra_comments: matches!(
            tcx.sess.opts.unstable_opts.mir_include_spans,
            MirIncludeSpans::On | MirIncludeSpans::Nll
        ),
    };

    let dumper = dumper.set_extra_data(extra_data).set_options(options);

    let _: io::Result<()> = try {
        let mut file = dumper.create_dump_file("html", body)?;
        emit_polonius_dump(
            &dumper,
            body,
            regioncx,
            borrow_set,
            &polonius_diagnostics.localized_outlives_constraints,
            &mut file,
        )?;
    };
}

/// The polonius dump consists of:
/// - the NLL MIR
/// - the list of polonius localized constraints
/// - a mermaid graph of the CFG
/// - a mermaid graph of the NLL regions and the constraints between them
/// - a mermaid graph of the NLL SCCs and the constraints between them
fn emit_polonius_dump<'tcx>(
    dumper: &MirDumper<'_, '_, 'tcx>,
    body: &Body<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
    localized_outlives_constraints: &LocalizedOutlivesConstraintSet,
    out: &mut dyn io::Write,
) -> io::Result<()> {
    // Prepare the HTML dump file prologue.
    writeln!(out, "<!DOCTYPE html>")?;
    writeln!(out, "<html>")?;
    writeln!(out, "<head><title>Polonius MIR dump</title></head>")?;
    writeln!(out, "<body>")?;

    // Section 1: the NLL + Polonius MIR.
    writeln!(out, "<div>")?;
    writeln!(out, "Raw MIR dump")?;
    writeln!(out, "<pre><code>")?;
    emit_html_mir(dumper, body, out)?;
    writeln!(out, "</code></pre>")?;
    writeln!(out, "</div>")?;

    // Section 2: mermaid visualization of the polonius constraint graph.
    writeln!(out, "<div>")?;
    writeln!(out, "Polonius constraint graph")?;
    writeln!(out, "<pre class='mermaid'>")?;
    let edge_count = emit_mermaid_constraint_graph(
        borrow_set,
        regioncx.liveness_constraints(),
        &localized_outlives_constraints,
        out,
    )?;
    writeln!(out, "</pre>")?;
    writeln!(out, "</div>")?;

    // Section 3: mermaid visualization of the CFG.
    writeln!(out, "<div>")?;
    writeln!(out, "Control-flow graph")?;
    writeln!(out, "<pre class='mermaid'>")?;
    emit_mermaid_cfg(body, out)?;
    writeln!(out, "</pre>")?;
    writeln!(out, "</div>")?;

    // Section 4: mermaid visualization of the NLL region graph.
    writeln!(out, "<div>")?;
    writeln!(out, "NLL regions")?;
    writeln!(out, "<pre class='mermaid'>")?;
    emit_mermaid_nll_regions(dumper.tcx(), regioncx, out)?;
    writeln!(out, "</pre>")?;
    writeln!(out, "</div>")?;

    // Section 5: mermaid visualization of the NLL SCC graph.
    writeln!(out, "<div>")?;
    writeln!(out, "NLL SCCs")?;
    writeln!(out, "<pre class='mermaid'>")?;
    emit_mermaid_nll_sccs(dumper.tcx(), regioncx, out)?;
    writeln!(out, "</pre>")?;
    writeln!(out, "</div>")?;

    // Finalize the dump with the HTML epilogue.
    writeln!(
        out,
        "<script src='https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js'></script>"
    )?;
    writeln!(out, "<script>")?;
    writeln!(
        out,
        "mermaid.initialize({{ startOnLoad: false, maxEdges: {} }});",
        edge_count.max(100),
    )?;
    writeln!(out, "mermaid.run({{ querySelector: '.mermaid' }})")?;
    writeln!(out, "</script>")?;
    writeln!(out, "</body>")?;
    writeln!(out, "</html>")?;

    Ok(())
}

/// Emits the polonius MIR, as escaped HTML.
fn emit_html_mir<'tcx>(
    dumper: &MirDumper<'_, '_, 'tcx>,
    body: &Body<'tcx>,
    out: &mut dyn io::Write,
) -> io::Result<()> {
    // Buffer the regular MIR dump to be able to escape it.
    let mut buffer = Vec::new();

    dumper.dump_mir_to_writer(body, &mut buffer)?;

    // Escape the handful of characters that need it. We don't need to be particularly efficient:
    // we're actually writing into a buffered writer already. Note that MIR dumps are valid UTF-8.
    let buffer = String::from_utf8_lossy(&buffer);
    for ch in buffer.chars() {
        let escaped = match ch {
            '>' => "&gt;",
            '<' => "&lt;",
            '&' => "&amp;",
            '\'' => "&#39;",
            '"' => "&quot;",
            _ => {
                // The common case, no escaping needed.
                write!(out, "{}", ch)?;
                continue;
            }
        };
        write!(out, "{}", escaped)?;
    }
    Ok(())
}

/// Produces the actual NLL + Polonius MIR sections to emit during the dumping process.
fn emit_polonius_mir<'tcx>(
    tcx: TyCtxt<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    closure_region_requirements: &Option<ClosureRegionRequirements<'tcx>>,
    borrow_set: &BorrowSet<'tcx>,
    localized_outlives_constraints: &LocalizedOutlivesConstraintSet,
    pass_where: PassWhere,
    out: &mut dyn io::Write,
) -> io::Result<()> {
    // Emit the regular NLL front-matter
    crate::nll::emit_nll_mir(
        tcx,
        regioncx,
        closure_region_requirements,
        borrow_set,
        pass_where,
        out,
    )?;

    let liveness = regioncx.liveness_constraints();

    // Add localized outlives constraints
    match pass_where {
        PassWhere::BeforeCFG => {
            if localized_outlives_constraints.outlives.len() > 0 {
                writeln!(out, "| Localized constraints")?;

                for constraint in &localized_outlives_constraints.outlives {
                    let LocalizedOutlivesConstraint { source, from, target, to } = constraint;
                    let from = liveness.location_from_point(*from);
                    let to = liveness.location_from_point(*to);
                    writeln!(out, "| {source:?} at {from:?} -> {target:?} at {to:?}")?;
                }
                writeln!(out, "|")?;
            }
        }
        _ => {}
    }

    Ok(())
}

/// Emits a mermaid flowchart of the CFG blocks and edges, similar to the graphviz version.
fn emit_mermaid_cfg(body: &Body<'_>, out: &mut dyn io::Write) -> io::Result<()> {
    use rustc_middle::mir::{TerminatorEdges, TerminatorKind};

    // The mermaid chart type: a top-down flowchart.
    writeln!(out, "flowchart TD")?;

    // Emit the block nodes.
    for (block_idx, block) in body.basic_blocks.iter_enumerated() {
        let block_idx = block_idx.as_usize();
        let cleanup = if block.is_cleanup { " (cleanup)" } else { "" };
        writeln!(out, "{block_idx}[\"bb{block_idx}{cleanup}\"]")?;
    }

    // Emit the edges between blocks, from the terminator edges.
    for (block_idx, block) in body.basic_blocks.iter_enumerated() {
        let block_idx = block_idx.as_usize();
        let terminator = block.terminator();
        match terminator.edges() {
            TerminatorEdges::None => {}
            TerminatorEdges::Single(bb) => {
                writeln!(out, "{block_idx} --> {}", bb.as_usize())?;
            }
            TerminatorEdges::Double(bb1, bb2) => {
                if matches!(terminator.kind, TerminatorKind::FalseEdge { .. }) {
                    writeln!(out, "{block_idx} --> {}", bb1.as_usize())?;
                    writeln!(out, "{block_idx} -- imaginary --> {}", bb2.as_usize())?;
                } else {
                    writeln!(out, "{block_idx} --> {}", bb1.as_usize())?;
                    writeln!(out, "{block_idx} -- unwind --> {}", bb2.as_usize())?;
                }
            }
            TerminatorEdges::AssignOnReturn { return_, cleanup, .. } => {
                for to_idx in return_ {
                    writeln!(out, "{block_idx} --> {}", to_idx.as_usize())?;
                }

                if let Some(to_idx) = cleanup {
                    writeln!(out, "{block_idx} -- unwind --> {}", to_idx.as_usize())?;
                }
            }
            TerminatorEdges::SwitchInt { targets, .. } => {
                for to_idx in targets.all_targets() {
                    writeln!(out, "{block_idx} --> {}", to_idx.as_usize())?;
                }
            }
        }
    }

    Ok(())
}

/// Emits a region's label: index, universe, external name.
fn render_region<'tcx>(
    tcx: TyCtxt<'tcx>,
    region: RegionVid,
    regioncx: &RegionInferenceContext<'tcx>,
    out: &mut dyn io::Write,
) -> io::Result<()> {
    let def = regioncx.region_definition(region);
    let universe = def.universe;

    write!(out, "'{}", region.as_usize())?;
    if !universe.is_root() {
        write!(out, "/{universe:?}")?;
    }
    if let Some(name) = def.external_name.and_then(|e| e.get_name(tcx)) {
        write!(out, " ({name})")?;
    }
    Ok(())
}

/// Emits a mermaid flowchart of the NLL regions and the outlives constraints between them, similar
/// to the graphviz version.
fn emit_mermaid_nll_regions<'tcx>(
    tcx: TyCtxt<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    out: &mut dyn io::Write,
) -> io::Result<()> {
    // The mermaid chart type: a top-down flowchart.
    writeln!(out, "flowchart TD")?;

    // Emit the region nodes.
    for region in regioncx.definitions.indices() {
        write!(out, "{}[\"", region.as_usize())?;
        render_region(tcx, region, regioncx, out)?;
        writeln!(out, "\"]")?;
    }

    // Get a set of edges to check for the reverse edge being present.
    let edges: FxHashSet<_> = regioncx.outlives_constraints().map(|c| (c.sup, c.sub)).collect();

    // Order (and deduplicate) edges for traversal, to display them in a generally increasing order.
    let constraint_key = |c: &OutlivesConstraint<'_>| {
        let min = c.sup.min(c.sub);
        let max = c.sup.max(c.sub);
        (min, max)
    };
    let mut ordered_edges: Vec<_> = regioncx.outlives_constraints().collect();
    ordered_edges.sort_by_key(|c| constraint_key(c));
    ordered_edges.dedup_by_key(|c| constraint_key(c));

    for outlives in ordered_edges {
        // Source node.
        write!(out, "{} ", outlives.sup.as_usize())?;

        // The kind of arrow: bidirectional if the opposite edge exists in the set.
        if edges.contains(&(outlives.sub, outlives.sup)) {
            write!(out, "&lt;")?;
        }
        write!(out, "-- ")?;

        // Edge label from its `Locations`.
        match outlives.locations {
            Locations::All(_) => write!(out, "All")?,
            Locations::Single(location) => write!(out, "{:?}", location)?,
        }

        // Target node.
        writeln!(out, " --> {}", outlives.sub.as_usize())?;
    }
    Ok(())
}

/// Emits a mermaid flowchart of the NLL SCCs and the outlives constraints between them, similar
/// to the graphviz version.
fn emit_mermaid_nll_sccs<'tcx>(
    tcx: TyCtxt<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    out: &mut dyn io::Write,
) -> io::Result<()> {
    // The mermaid chart type: a top-down flowchart.
    writeln!(out, "flowchart TD")?;

    // Gather and emit the SCC nodes.
    let mut nodes_per_scc: IndexVec<_, _> =
        regioncx.constraint_sccs().all_sccs().map(|_| Vec::new()).collect();
    for region in regioncx.definitions.indices() {
        let scc = regioncx.constraint_sccs().scc(region);
        nodes_per_scc[scc].push(region);
    }
    for (scc, regions) in nodes_per_scc.iter_enumerated() {
        // The node label: the regions contained in the SCC.
        write!(out, "{scc}[\"SCC({scc}) = {{", scc = scc.as_usize())?;
        for (idx, &region) in regions.iter().enumerate() {
            render_region(tcx, region, regioncx, out)?;
            if idx < regions.len() - 1 {
                write!(out, ",")?;
            }
        }
        writeln!(out, "}}\"]")?;
    }

    // Emit the edges between SCCs.
    let edges = regioncx.constraint_sccs().all_sccs().flat_map(|source| {
        regioncx.constraint_sccs().successors(source).iter().map(move |&target| (source, target))
    });
    for (source, target) in edges {
        writeln!(out, "{} --> {}", source.as_usize(), target.as_usize())?;
    }

    Ok(())
}

/// Emits a mermaid flowchart of the polonius localized outlives constraints, with subgraphs per
/// region, and loan introductions.
fn emit_mermaid_constraint_graph<'tcx>(
    borrow_set: &BorrowSet<'tcx>,
    liveness: &LivenessValues,
    localized_outlives_constraints: &LocalizedOutlivesConstraintSet,
    out: &mut dyn io::Write,
) -> io::Result<usize> {
    let location_name = |location: Location| {
        // A MIR location looks like `bb5[2]`. As that is not a syntactically valid mermaid node id,
        // transform it into `BB5_2`.
        format!("BB{}_{}", location.block.index(), location.statement_index)
    };
    let region_name = |region: RegionVid| format!("'{}", region.index());
    let node_name = |region: RegionVid, point: PointIndex| {
        let location = liveness.location_from_point(point);
        format!("{}_{}", region_name(region), location_name(location))
    };

    // The mermaid chart type: a top-down flowchart, which supports subgraphs.
    writeln!(out, "flowchart TD")?;

    // The loans subgraph: a node per loan.
    writeln!(out, "    subgraph \"Loans\"")?;
    for loan_idx in 0..borrow_set.len() {
        writeln!(out, "        L{loan_idx}")?;
    }
    writeln!(out, "    end\n")?;

    // And an edge from that loan node to where it enters the constraint graph.
    for (loan_idx, loan) in borrow_set.iter_enumerated() {
        writeln!(
            out,
            "    L{} --> {}_{}",
            loan_idx.index(),
            region_name(loan.region),
            location_name(loan.reserve_location),
        )?;
    }
    writeln!(out, "")?;

    // The regions subgraphs containing the region/point nodes.
    let mut points_per_region: FxIndexMap<RegionVid, FxIndexSet<PointIndex>> =
        FxIndexMap::default();
    for constraint in &localized_outlives_constraints.outlives {
        points_per_region.entry(constraint.source).or_default().insert(constraint.from);
        points_per_region.entry(constraint.target).or_default().insert(constraint.to);
    }
    for (region, points) in points_per_region {
        writeln!(out, "    subgraph \"{}\"", region_name(region))?;
        for point in points {
            writeln!(out, "        {}", node_name(region, point))?;
        }
        writeln!(out, "    end\n")?;
    }

    // The constraint graph edges.
    for constraint in &localized_outlives_constraints.outlives {
        // FIXME: add killed loans and constraint kind as edge labels.
        writeln!(
            out,
            "    {} --> {}",
            node_name(constraint.source, constraint.from),
            node_name(constraint.target, constraint.to),
        )?;
    }

    // Return the number of edges: this is the biggest graph in the dump and its edge count will be
    // mermaid's max edge count to support.
    let edge_count = borrow_set.len() + localized_outlives_constraints.outlives.len();
    Ok(edge_count)
}
