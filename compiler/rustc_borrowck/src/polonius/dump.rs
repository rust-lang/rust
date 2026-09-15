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
use crate::dataflow::BorrowIndex;
use crate::polonius::{LocalizedConstraintGraphVisitor, LocalizedNode, PoloniusContext};
use crate::region_infer::values::LivenessValues;
use crate::type_check::Locations;
use crate::{BorrowckInferCtxt, ClosureRegionRequirements, RegionInferenceContext};

/// The polonius MIR dump template: a regular HTML file for easy editing, with special dummy
/// sections to be replaced by real contents.
const TEMPLATE: &str = include_str!("./dump/polonius-mir-dump.template.html");

/// `-Zdump-mir=polonius` dumps MIR annotated with NLL and polonius specific information.
pub(crate) fn dump_polonius_mir<'tcx>(
    infcx: &BorrowckInferCtxt<'tcx>,
    body: &Body<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    closure_region_requirements: &Option<ClosureRegionRequirements<'tcx>>,
    borrow_set: &BorrowSet<'tcx>,
    polonius_context: Option<&PoloniusContext>,
) {
    let tcx = infcx.tcx;
    if !tcx.sess.opts.unstable_opts.polonius.is_next_enabled() {
        return;
    }

    let Some(dumper) = MirDumper::new(tcx, "polonius", body) else { return };

    let polonius_context =
        polonius_context.expect("missing polonius context with `-Zpolonius=next`");

    // If we have a polonius graph to dump along the rest of the MIR and NLL info, we extract its
    // constraints here.
    let mut collector = MirDumpCollector::default();
    if let Some(graph) = &polonius_context.graph {
        graph.traverse(
            body,
            regioncx.liveness_constraints(),
            &polonius_context.live_region_variances,
            regioncx.universal_regions(),
            borrow_set,
            &mut collector,
        );
    }

    let extra_data = &|pass_where, out: &mut dyn io::Write| {
        emit_polonius_mir(
            tcx,
            regioncx,
            closure_region_requirements,
            borrow_set,
            &collector.constraints,
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

    let _ = try {
        let mut file = dumper.create_dump_file("html", body)?;
        emit_polonius_dump(&dumper, body, regioncx, borrow_set, &collector, &mut file)?;
    };
}

/// The constraints we'll dump as text or a mermaid graph.
struct LocalizedOutlivesConstraint {
    source: RegionVid,
    from: PointIndex,
    target: RegionVid,
    to: PointIndex,
}

/// Visitor to record constraints encountered when traversing the localized constraint graph, as
/// well as the reachability of each loan.
#[derive(Default)]
struct MirDumpCollector {
    constraints: Vec<LocalizedOutlivesConstraint>,
    reachability: FxIndexMap<BorrowIndex, Vec<LocalizedNode>>,
}

impl LocalizedConstraintGraphVisitor for MirDumpCollector {
    fn on_node_traversed(&mut self, loan: BorrowIndex, node: LocalizedNode) {
        self.reachability.entry(loan).or_default().push(node);
    }

    fn on_successor_discovered(&mut self, current_node: LocalizedNode, successor: LocalizedNode) {
        self.constraints.push(LocalizedOutlivesConstraint {
            source: current_node.region,
            from: current_node.point,
            target: successor.region,
            to: successor.point,
        });
    }
}

/// The polonius dump consists of:
/// - the NLL MIR
/// - the list of polonius localized constraints
/// - a mermaid graph of the CFG
/// - a mermaid graph of the NLL regions and the constraints between them
/// - a mermaid graph of the NLL SCCs and the constraints between them
fn emit_polonius_dump<'tcx>(
    dumper: &MirDumper<'_, 'tcx>,
    body: &Body<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
    collector: &MirDumpCollector,
    out: &mut dyn io::Write,
) -> io::Result<()> {
    let mut edge_count = 0;

    // We replace the dummy $SECTION tokens from the HTML polonius dump template, and emit the
    // result into the given writer.
    for chunk in TEMPLATE.split("$SECTION") {
        match chunk.strip_prefix("_") {
            None => {
                // We're at the beginning of the template: this is the prologue to emit as-is.
                writeln!(out, "{}", chunk)?;
            }
            Some(section) => {
                // This is the start of a prefixed section, we look for its identifier.
                let dummy_section_end = section
                    .find("<")
                    .expect("the template section end boundary needs to be present");
                let section_identifier = section[..dummy_section_end].trim();

                // Emit the real section instead of the dummy token.
                match section_identifier {
                    "MIR" => {
                        emit_html_mir(dumper, body, out)?;
                    }
                    "POLONIUS_CONSTRAINTS" => {
                        edge_count = emit_mermaid_constraint_graph(
                            borrow_set,
                            regioncx.liveness_constraints(),
                            &collector.constraints,
                            out,
                        )?;
                    }
                    "POLONIUS_REACHABILITY" => {
                        emit_loan_reachability(
                            borrow_set,
                            regioncx.liveness_constraints(),
                            &collector.reachability,
                            out,
                        )?;
                    }
                    "CFG" => {
                        emit_mermaid_cfg(body, out)?;
                    }
                    "NLL_CONSTRAINTS" => {
                        emit_mermaid_nll_regions(dumper.tcx(), regioncx, out)?;
                    }
                    "NLL_SCCS" => {
                        emit_mermaid_nll_sccs(dumper.tcx(), regioncx, out)?;
                    }
                    "INITIALIZATION" => {
                        writeln!(out, "<script>")?;
                        writeln!(
                            out,
                            "mermaid.initialize({{ startOnLoad: false, maxEdges: {} }});",
                            edge_count.max(100),
                        )?;
                        writeln!(out, "mermaid.run({{ querySelector: '.mermaid' }})")?;
                        writeln!(out, "</script>")?;
                    }

                    _ => {
                        unreachable!("unexpected dummy section identifier {:?}", section_identifier)
                    }
                }

                // And finally, emit the contents that followed the dummy token.
                writeln!(out, "{}", &section[dummy_section_end..])?;
            }
        }
    }

    Ok(())
}

/// Emits the polonius MIR, as escaped HTML.
fn emit_html_mir<'tcx>(
    dumper: &MirDumper<'_, 'tcx>,
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
    localized_outlives_constraints: &[LocalizedOutlivesConstraint],
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
            if localized_outlives_constraints.len() > 0 {
                writeln!(out, "| Localized constraints")?;

                for constraint in localized_outlives_constraints {
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
    localized_outlives_constraints: &[LocalizedOutlivesConstraint],
    out: &mut dyn io::Write,
) -> io::Result<usize> {
    let node_label = |region: RegionVid, point: PointIndex| {
        let location = liveness.location_from_point(point);
        node_name(region, location)
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
    for constraint in localized_outlives_constraints {
        points_per_region.entry(constraint.source).or_default().insert(constraint.from);
        points_per_region.entry(constraint.target).or_default().insert(constraint.to);
    }
    for (region, points) in points_per_region {
        writeln!(out, "    subgraph \"{}\"", region_name(region))?;
        for point in points {
            writeln!(out, "        {}", node_label(region, point))?;
        }
        writeln!(out, "    end\n")?;
    }

    // The constraint graph edges.
    for constraint in localized_outlives_constraints {
        // FIXME: add killed loans and constraint kind as edge labels.
        writeln!(
            out,
            "    {} --> {}",
            node_label(constraint.source, constraint.from),
            node_label(constraint.target, constraint.to),
        )?;
    }

    // Return the number of edges: this is the biggest graph in the dump and its edge count will be
    // mermaid's max edge count to support.
    let edge_count = borrow_set.len() + localized_outlives_constraints.len();
    Ok(edge_count)
}

/// Emits the reachability of loans: a list of all nodes reached while traversing the polonius
/// constraint graph.
fn emit_loan_reachability(
    borrow_set: &BorrowSet<'_>,
    liveness: &LivenessValues,
    reachability: &FxIndexMap<BorrowIndex, Vec<LocalizedNode>>,
    out: &mut dyn io::Write,
) -> io::Result<()> {
    for (loan, _) in borrow_set.iter_enumerated() {
        let Some(reachability) = reachability.get(&loan) else {
            continue;
        };
        let loan = format!("L{}", loan.index());

        // The button to display the loan trace. The javascript event listener is hooked up in the
        // template itself.
        writeln!(
            out,
            "<div class='trace'><button data-loan='{loan}'>Trace for loan {loan}</button></div>"
        )?;

        // The actual trace contents, hidden by default.
        writeln!(out, "<div id='trace-{loan}' class='trace hidden'>")?;
        writeln!(out, "<div>Trace for loan {loan}</div>")?;
        writeln!(out, "<ul>")?;
        for (idx, node) in reachability.iter().enumerate() {
            writeln!(out, "<li>")?;

            let location = liveness.location_from_point(node.point);
            let kind = if idx == 0 { "starts in" } else { "reaches" };
            writeln!(
                out,
                "<code>{loan}</code> {kind} <code>{}</code>",
                node_name(node.region, location),
            )?;

            // It's useful to know whether the region we're reaching is live at this point.
            let node_liveness =
                if liveness.is_live_at(node.region, location) { "live" } else { "not live" };
            writeln!(
                out,
                "/ at <code>{:?}</code>: <code>'{}</code> is {}",
                location,
                node.region.index(),
                node_liveness,
            )?;
            writeln!(out, "</li>")?;
        }
        writeln!(out, "</ul>")?;
        writeln!(out, "</div>")?;
    }

    Ok(())
}

fn region_name(region: RegionVid) -> String {
    format!("'{}", region.index())
}
/// A MIR location looks like `bb5[2]`. As that is not a syntactically valid mermaid node id,
/// transform it into `BB5_2`.
fn location_name(location: Location) -> String {
    format!("BB{}_{}", location.block.index(), location.statement_index)
}
fn node_name(region: RegionVid, location: Location) -> String {
    format!("{}_{}", region_name(region), location_name(location))
}
