//! The entry point of the NLL borrow checker.

use rustc_data_structures::vec_map::VecMap;
use rustc_errors::Diagnostic;
use rustc_index::vec::IndexVec;
use rustc_infer::infer::InferCtxt;
use rustc_middle::mir::{create_dump_file, dump_enabled, dump_mir, PassWhere};
use rustc_middle::mir::{
    BasicBlock, Body, ClosureOutlivesSubject, ClosureRegionRequirements, LocalKind, Location,
    Promoted,
};
use rustc_middle::ty::{self, OpaqueTypeKey, RegionKind, RegionVid, Ty};
use rustc_span::symbol::sym;
use std::env;
use std::fmt::Debug;
use std::io;
use std::path::PathBuf;
use std::rc::Rc;
use std::str::FromStr;

use polonius_engine::{Algorithm, Output};

use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::{InitKind, InitLocation, MoveData};
use rustc_mir_dataflow::ResultsCursor;

use crate::{
    borrow_set::BorrowSet,
    constraint_generation,
    diagnostics::RegionErrors,
    facts::{AllFacts, AllFactsExt, RustcFacts},
    invalidation,
    location::LocationTable,
    region_infer::{values::RegionValueElements, RegionInferenceContext},
    renumber,
    type_check::{self, MirTypeckRegionConstraints, MirTypeckResults},
    universal_regions::UniversalRegions,
    Upvar,
};

pub type PoloniusOutput = Output<RustcFacts>;

/// The output of `nll::compute_regions`. This includes the computed `RegionInferenceContext`, any
/// closure requirements to propagate, and any generated errors.
crate struct NllOutput<'tcx> {
    pub regioncx: RegionInferenceContext<'tcx>,
    pub opaque_type_values: VecMap<OpaqueTypeKey<'tcx>, Ty<'tcx>>,
    pub polonius_input: Option<Box<AllFacts>>,
    pub polonius_output: Option<Rc<PoloniusOutput>>,
    pub opt_closure_req: Option<ClosureRegionRequirements<'tcx>>,
    pub nll_errors: RegionErrors<'tcx>,
}

/// Rewrites the regions in the MIR to use NLL variables, also scraping out the set of universal
/// regions (e.g., region parameters) declared on the function. That set will need to be given to
/// `compute_regions`.
pub(crate) fn replace_regions_in_mir<'cx, 'tcx>(
    infcx: &InferCtxt<'cx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body: &mut Body<'tcx>,
    promoted: &mut IndexVec<Promoted, Body<'tcx>>,
) -> UniversalRegions<'tcx> {
    let def = body.source.with_opt_param().as_local().unwrap();

    debug!("replace_regions_in_mir(def={:?})", def);

    // Compute named region information. This also renumbers the inputs/outputs.
    let universal_regions = UniversalRegions::new(infcx, def, param_env);

    // Replace all remaining regions with fresh inference variables.
    renumber::renumber_mir(infcx, body, promoted);

    dump_mir(infcx.tcx, None, "renumber", &0, body, |_, _| Ok(()));

    universal_regions
}

// This function populates an AllFacts instance with base facts related to
// MovePaths and needed for the move analysis.
fn populate_polonius_move_facts(
    all_facts: &mut AllFacts,
    move_data: &MoveData<'_>,
    location_table: &LocationTable,
    body: &Body<'_>,
) {
    all_facts
        .path_is_var
        .extend(move_data.rev_lookup.iter_locals_enumerated().map(|(v, &m)| (m, v)));

    for (child, move_path) in move_data.move_paths.iter_enumerated() {
        if let Some(parent) = move_path.parent {
            all_facts.child_path.push((child, parent));
        }
    }

    let fn_entry_start = location_table
        .start_index(Location { block: BasicBlock::from_u32(0u32), statement_index: 0 });

    // initialized_at
    for init in move_data.inits.iter() {
        match init.location {
            InitLocation::Statement(location) => {
                let block_data = &body[location.block];
                let is_terminator = location.statement_index == block_data.statements.len();

                if is_terminator && init.kind == InitKind::NonPanicPathOnly {
                    // We are at the terminator of an init that has a panic path,
                    // and where the init should not happen on panic

                    for &successor in block_data.terminator().successors() {
                        if body[successor].is_cleanup {
                            continue;
                        }

                        // The initialization happened in (or rather, when arriving at)
                        // the successors, but not in the unwind block.
                        let first_statement = Location { block: successor, statement_index: 0 };
                        all_facts
                            .path_assigned_at_base
                            .push((init.path, location_table.start_index(first_statement)));
                    }
                } else {
                    // In all other cases, the initialization just happens at the
                    // midpoint, like any other effect.
                    all_facts
                        .path_assigned_at_base
                        .push((init.path, location_table.mid_index(location)));
                }
            }
            // Arguments are initialized on function entry
            InitLocation::Argument(local) => {
                assert!(body.local_kind(local) == LocalKind::Arg);
                all_facts.path_assigned_at_base.push((init.path, fn_entry_start));
            }
        }
    }

    for (local, &path) in move_data.rev_lookup.iter_locals_enumerated() {
        if body.local_kind(local) != LocalKind::Arg {
            // Non-arguments start out deinitialised; we simulate this with an
            // initial move:
            all_facts.path_moved_at_base.push((path, fn_entry_start));
        }
    }

    // moved_out_at
    // deinitialisation is assumed to always happen!
    all_facts
        .path_moved_at_base
        .extend(move_data.moves.iter().map(|mo| (mo.path, location_table.mid_index(mo.source))));
}

/// Computes the (non-lexical) regions from the input MIR.
///
/// This may result in errors being reported.
pub(crate) fn compute_regions<'cx, 'tcx>(
    infcx: &InferCtxt<'cx, 'tcx>,
    universal_regions: UniversalRegions<'tcx>,
    body: &Body<'tcx>,
    promoted: &IndexVec<Promoted, Body<'tcx>>,
    location_table: &LocationTable,
    param_env: ty::ParamEnv<'tcx>,
    flow_inits: &mut ResultsCursor<'cx, 'tcx, MaybeInitializedPlaces<'cx, 'tcx>>,
    move_data: &MoveData<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
    upvars: &[Upvar<'tcx>],
    use_polonius: bool,
) -> NllOutput<'tcx> {
    let mut all_facts =
        (use_polonius || AllFacts::enabled(infcx.tcx)).then_some(AllFacts::default());

    let universal_regions = Rc::new(universal_regions);

    let elements = &Rc::new(RegionValueElements::new(&body));

    // Run the MIR type-checker.
    let MirTypeckResults { constraints, universal_region_relations, opaque_type_values } =
        type_check::type_check(
            infcx,
            param_env,
            body,
            promoted,
            &universal_regions,
            location_table,
            borrow_set,
            &mut all_facts,
            flow_inits,
            move_data,
            elements,
            upvars,
        );

    if let Some(all_facts) = &mut all_facts {
        let _prof_timer = infcx.tcx.prof.generic_activity("polonius_fact_generation");
        all_facts.universal_region.extend(universal_regions.universal_regions());
        populate_polonius_move_facts(all_facts, move_data, location_table, &body);

        // Emit universal regions facts, and their relations, for Polonius.
        //
        // 1: universal regions are modeled in Polonius as a pair:
        // - the universal region vid itself.
        // - a "placeholder loan" associated to this universal region. Since they don't exist in
        //   the `borrow_set`, their `BorrowIndex` are synthesized as the universal region index
        //   added to the existing number of loans, as if they succeeded them in the set.
        //
        let borrow_count = borrow_set.len();
        debug!(
            "compute_regions: polonius placeholders, num_universals={}, borrow_count={}",
            universal_regions.len(),
            borrow_count
        );

        for universal_region in universal_regions.universal_regions() {
            let universal_region_idx = universal_region.index();
            let placeholder_loan_idx = borrow_count + universal_region_idx;
            all_facts.placeholder.push((universal_region, placeholder_loan_idx.into()));
        }

        // 2: the universal region relations `outlives` constraints are emitted as
        //  `known_placeholder_subset` facts.
        for (fr1, fr2) in universal_region_relations.known_outlives() {
            if fr1 != fr2 {
                debug!(
                    "compute_regions: emitting polonius `known_placeholder_subset` \
                     fr1={:?}, fr2={:?}",
                    fr1, fr2
                );
                all_facts.known_placeholder_subset.push((*fr1, *fr2));
            }
        }
    }

    // Create the region inference context, taking ownership of the
    // region inference data that was contained in `infcx`, and the
    // base constraints generated by the type-check.
    let var_origins = infcx.take_region_var_origins();
    let MirTypeckRegionConstraints {
        placeholder_indices,
        placeholder_index_to_region: _,
        mut liveness_constraints,
        outlives_constraints,
        member_constraints,
        closure_bounds_mapping,
        universe_causes,
        type_tests,
    } = constraints;
    let placeholder_indices = Rc::new(placeholder_indices);

    constraint_generation::generate_constraints(
        infcx,
        &mut liveness_constraints,
        &mut all_facts,
        location_table,
        &body,
        borrow_set,
    );

    let mut regioncx = RegionInferenceContext::new(
        var_origins,
        universal_regions,
        placeholder_indices,
        universal_region_relations,
        outlives_constraints,
        member_constraints,
        closure_bounds_mapping,
        universe_causes,
        type_tests,
        liveness_constraints,
        elements,
    );

    // Generate various additional constraints.
    invalidation::generate_invalidates(infcx.tcx, &mut all_facts, location_table, body, borrow_set);

    let def_id = body.source.def_id();

    // Dump facts if requested.
    let polonius_output = all_facts.as_ref().and_then(|all_facts| {
        if infcx.tcx.sess.opts.debugging_opts.nll_facts {
            let def_path = infcx.tcx.def_path(def_id);
            let dir_path = PathBuf::from(&infcx.tcx.sess.opts.debugging_opts.nll_facts_dir)
                .join(def_path.to_filename_friendly_no_crate());
            all_facts.write_to_dir(dir_path, location_table).unwrap();
        }

        if use_polonius {
            let algorithm =
                env::var("POLONIUS_ALGORITHM").unwrap_or_else(|_| String::from("Hybrid"));
            let algorithm = Algorithm::from_str(&algorithm).unwrap();
            debug!("compute_regions: using polonius algorithm {:?}", algorithm);
            let _prof_timer = infcx.tcx.prof.generic_activity("polonius_analysis");
            Some(Rc::new(Output::compute(&all_facts, algorithm, false)))
        } else {
            None
        }
    });

    // Solve the region constraints.
    let (closure_region_requirements, nll_errors) =
        regioncx.solve(infcx, &body, polonius_output.clone());

    if !nll_errors.is_empty() {
        // Suppress unhelpful extra errors in `infer_opaque_types`.
        infcx.set_tainted_by_errors();
    }

    let remapped_opaque_tys = regioncx.infer_opaque_types(&infcx, opaque_type_values, body.span);

    NllOutput {
        regioncx,
        opaque_type_values: remapped_opaque_tys,
        polonius_input: all_facts.map(Box::new),
        polonius_output,
        opt_closure_req: closure_region_requirements,
        nll_errors,
    }
}

pub(super) fn dump_mir_results<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    body: &Body<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    closure_region_requirements: &Option<ClosureRegionRequirements<'_>>,
) {
    if !dump_enabled(infcx.tcx, "nll", body.source.def_id()) {
        return;
    }

    dump_mir(infcx.tcx, None, "nll", &0, body, |pass_where, out| {
        match pass_where {
            // Before the CFG, dump out the values for each region variable.
            PassWhere::BeforeCFG => {
                regioncx.dump_mir(infcx.tcx, out)?;
                writeln!(out, "|")?;

                if let Some(closure_region_requirements) = closure_region_requirements {
                    writeln!(out, "| Free Region Constraints")?;
                    for_each_region_constraint(closure_region_requirements, &mut |msg| {
                        writeln!(out, "| {}", msg)
                    })?;
                    writeln!(out, "|")?;
                }
            }

            PassWhere::BeforeLocation(_) => {}

            PassWhere::AfterTerminator(_) => {}

            PassWhere::BeforeBlock(_) | PassWhere::AfterLocation(_) | PassWhere::AfterCFG => {}
        }
        Ok(())
    });

    // Also dump the inference graph constraints as a graphviz file.
    let _: io::Result<()> = try {
        let mut file =
            create_dump_file(infcx.tcx, "regioncx.all.dot", None, "nll", &0, body.source)?;
        regioncx.dump_graphviz_raw_constraints(&mut file)?;
    };

    // Also dump the inference graph constraints as a graphviz file.
    let _: io::Result<()> = try {
        let mut file =
            create_dump_file(infcx.tcx, "regioncx.scc.dot", None, "nll", &0, body.source)?;
        regioncx.dump_graphviz_scc_constraints(&mut file)?;
    };
}

pub(super) fn dump_annotation<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    body: &Body<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    closure_region_requirements: &Option<ClosureRegionRequirements<'_>>,
    opaque_type_values: &VecMap<OpaqueTypeKey<'tcx>, Ty<'tcx>>,
    errors_buffer: &mut Vec<Diagnostic>,
) {
    let tcx = infcx.tcx;
    let base_def_id = tcx.closure_base_def_id(body.source.def_id());
    if !tcx.has_attr(base_def_id, sym::rustc_regions) {
        return;
    }

    // When the enclosing function is tagged with `#[rustc_regions]`,
    // we dump out various bits of state as warnings. This is useful
    // for verifying that the compiler is behaving as expected.  These
    // warnings focus on the closure region requirements -- for
    // viewing the intraprocedural state, the -Zdump-mir output is
    // better.

    let mut err = if let Some(closure_region_requirements) = closure_region_requirements {
        let mut err = tcx.sess.diagnostic().span_note_diag(body.span, "external requirements");

        regioncx.annotate(tcx, &mut err);

        err.note(&format!(
            "number of external vids: {}",
            closure_region_requirements.num_external_vids
        ));

        // Dump the region constraints we are imposing *between* those
        // newly created variables.
        for_each_region_constraint(closure_region_requirements, &mut |msg| {
            err.note(msg);
            Ok(())
        })
        .unwrap();

        err
    } else {
        let mut err = tcx.sess.diagnostic().span_note_diag(body.span, "no external requirements");
        regioncx.annotate(tcx, &mut err);

        err
    };

    if !opaque_type_values.is_empty() {
        err.note(&format!("Inferred opaque type values:\n{:#?}", opaque_type_values));
    }

    err.buffer(errors_buffer);
}

fn for_each_region_constraint(
    closure_region_requirements: &ClosureRegionRequirements<'_>,
    with_msg: &mut dyn FnMut(&str) -> io::Result<()>,
) -> io::Result<()> {
    for req in &closure_region_requirements.outlives_requirements {
        let subject: &dyn Debug = match &req.subject {
            ClosureOutlivesSubject::Region(subject) => subject,
            ClosureOutlivesSubject::Ty(ty) => ty,
        };
        with_msg(&format!("where {:?}: {:?}", subject, req.outlived_free_region,))?;
    }
    Ok(())
}

/// Right now, we piggy back on the `ReVar` to store our NLL inference
/// regions. These are indexed with `RegionVid`. This method will
/// assert that the region is a `ReVar` and extract its internal index.
/// This is reasonable because in our MIR we replace all universal regions
/// with inference variables.
pub trait ToRegionVid {
    fn to_region_vid(self) -> RegionVid;
}

impl<'tcx> ToRegionVid for &'tcx RegionKind {
    fn to_region_vid(self) -> RegionVid {
        if let ty::ReVar(vid) = self { *vid } else { bug!("region is not an ReVar: {:?}", self) }
    }
}

impl ToRegionVid for RegionVid {
    fn to_region_vid(self) -> RegionVid {
        self
    }
}

crate trait ConstraintDescription {
    fn description(&self) -> &'static str;
}
