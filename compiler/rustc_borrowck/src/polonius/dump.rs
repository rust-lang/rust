use std::io;

use rustc_middle::mir::pretty::{
    PassWhere, PrettyPrintMirOptions, create_dump_file, dump_enabled, dump_mir_to_writer,
};
use rustc_middle::mir::{Body, ClosureRegionRequirements};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::MirIncludeSpans;

use crate::borrow_set::BorrowSet;
use crate::polonius::{LocalizedOutlivesConstraint, LocalizedOutlivesConstraintSet};
use crate::{BorrowckInferCtxt, RegionInferenceContext};

/// `-Zdump-mir=polonius` dumps MIR annotated with NLL and polonius specific information.
pub(crate) fn dump_polonius_mir<'tcx>(
    infcx: &BorrowckInferCtxt<'tcx>,
    body: &Body<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
    localized_outlives_constraints: Option<LocalizedOutlivesConstraintSet>,
    closure_region_requirements: &Option<ClosureRegionRequirements<'tcx>>,
) {
    let tcx = infcx.tcx;
    if !tcx.sess.opts.unstable_opts.polonius.is_next_enabled() {
        return;
    }

    if !dump_enabled(tcx, "polonius", body.source.def_id()) {
        return;
    }

    let localized_outlives_constraints = localized_outlives_constraints
        .expect("missing localized constraints with `-Zpolonius=next`");

    let _: io::Result<()> = try {
        let mut file = create_dump_file(tcx, "html", false, "polonius", &0, body)?;
        emit_polonius_dump(
            tcx,
            body,
            regioncx,
            borrow_set,
            localized_outlives_constraints,
            closure_region_requirements,
            &mut file,
        )?;
    };
}

/// The polonius dump consists of:
/// - the NLL MIR
/// - the list of polonius localized constraints
/// - a mermaid graph of the CFG
fn emit_polonius_dump<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
    localized_outlives_constraints: LocalizedOutlivesConstraintSet,
    closure_region_requirements: &Option<ClosureRegionRequirements<'tcx>>,
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
    writeln!(out, "<code><pre>")?;
    emit_html_mir(
        tcx,
        body,
        regioncx,
        borrow_set,
        localized_outlives_constraints,
        closure_region_requirements,
        out,
    )?;
    writeln!(out, "</pre></code>")?;
    writeln!(out, "</div>")?;

    // Section 2: mermaid visualization of the CFG.
    writeln!(out, "<div>")?;
    writeln!(out, "Control-flow graph")?;
    writeln!(out, "<code><pre class='mermaid'>")?;
    emit_mermaid_cfg(body, out)?;
    writeln!(out, "</pre></code>")?;
    writeln!(out, "</div>")?;

    // Finalize the dump with the HTML epilogue.
    writeln!(
        out,
        "<script src='https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js'></script>"
    )?;
    writeln!(out, "<script>")?;
    writeln!(out, "mermaid.initialize({{ startOnLoad: false, maxEdges: 100 }});")?;
    writeln!(out, "mermaid.run({{ querySelector: '.mermaid' }})")?;
    writeln!(out, "</script>")?;
    writeln!(out, "</body>")?;
    writeln!(out, "</html>")?;

    Ok(())
}

/// Emits the polonius MIR, as escaped HTML.
fn emit_html_mir<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
    localized_outlives_constraints: LocalizedOutlivesConstraintSet,
    closure_region_requirements: &Option<ClosureRegionRequirements<'tcx>>,
    out: &mut dyn io::Write,
) -> io::Result<()> {
    // Buffer the regular MIR dump to be able to escape it.
    let mut buffer = Vec::new();

    // We want the NLL extra comments printed by default in NLL MIR dumps. Specifying `-Z
    // mir-include-spans` on the CLI still has priority.
    let options = PrettyPrintMirOptions {
        include_extra_comments: matches!(
            tcx.sess.opts.unstable_opts.mir_include_spans,
            MirIncludeSpans::On | MirIncludeSpans::Nll
        ),
    };

    dump_mir_to_writer(
        tcx,
        "polonius",
        &0,
        body,
        &mut buffer,
        |pass_where, out| {
            emit_polonius_mir(
                tcx,
                regioncx,
                closure_region_requirements,
                borrow_set,
                &localized_outlives_constraints,
                pass_where,
                out,
            )
        },
        options,
    )?;

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
        pass_where.clone(),
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
