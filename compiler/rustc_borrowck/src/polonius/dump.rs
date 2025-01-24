use std::io;

use rustc_middle::mir::pretty::{PrettyPrintMirOptions, dump_mir_with_options};
use rustc_middle::mir::{Body, ClosureRegionRequirements, PassWhere};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::MirIncludeSpans;

use crate::borrow_set::BorrowSet;
use crate::polonius::{LocalizedOutlivesConstraint, LocalizedOutlivesConstraintSet};
use crate::{BorrowckInferCtxt, RegionInferenceContext};

/// `-Zdump-mir=polonius` dumps MIR annotated with NLL and polonius specific information.
// Note: this currently duplicates most of NLL MIR, with some additions for the localized outlives
// constraints. This is ok for now as this dump will change in the near future to an HTML file to
// become more useful.
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

    let localized_outlives_constraints = localized_outlives_constraints
        .expect("missing localized constraints with `-Zpolonius=next`");

    // We want the NLL extra comments printed by default in NLL MIR dumps (they were removed in
    // #112346). Specifying `-Z mir-include-spans` on the CLI still has priority: for example,
    // they're always disabled in mir-opt tests to make working with blessed dumps easier.
    let options = PrettyPrintMirOptions {
        include_extra_comments: matches!(
            tcx.sess.opts.unstable_opts.mir_include_spans,
            MirIncludeSpans::On | MirIncludeSpans::Nll
        ),
    };

    dump_mir_with_options(
        tcx,
        false,
        "polonius",
        &0,
        body,
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
    );
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
