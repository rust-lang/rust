use rustc_data_structures::captures::Captures;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::coverage::{
    CounterId, CovTerm, CoverageIdsInfo, CoverageKind, Expression, ExpressionId,
    FunctionCoverageInfo, MappingKind, Op,
};
use rustc_middle::mir::{Body, Statement, StatementKind};
use rustc_middle::ty::{self, TyCtxt};
use rustc_middle::util::Providers;
use rustc_span::def_id::LocalDefId;
use rustc_span::sym;
use tracing::trace;

/// Registers query/hook implementations related to coverage.
pub(crate) fn provide(providers: &mut Providers) {
    providers.hooks.is_eligible_for_coverage = is_eligible_for_coverage;
    providers.queries.coverage_attr_on = coverage_attr_on;
    providers.queries.coverage_ids_info = coverage_ids_info;
}

/// Hook implementation for [`TyCtxt::is_eligible_for_coverage`].
fn is_eligible_for_coverage(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    // Only instrument functions, methods, and closures (not constants since they are evaluated
    // at compile time by Miri).
    // FIXME(#73156): Handle source code coverage in const eval, but note, if and when const
    // expressions get coverage spans, we will probably have to "carve out" space for const
    // expressions from coverage spans in enclosing MIR's, like we do for closures. (That might
    // be tricky if const expressions have no corresponding statements in the enclosing MIR.
    // Closures are carved out by their initial `Assign` statement.)
    if !tcx.def_kind(def_id).is_fn_like() {
        trace!("InstrumentCoverage skipped for {def_id:?} (not an fn-like)");
        return false;
    }

    // Don't instrument functions with `#[automatically_derived]` on their
    // enclosing impl block, on the assumption that most users won't care about
    // coverage for derived impls.
    if let Some(impl_of) = tcx.impl_of_method(def_id.to_def_id())
        && tcx.is_automatically_derived(impl_of)
    {
        trace!("InstrumentCoverage skipped for {def_id:?} (automatically derived)");
        return false;
    }

    if tcx.codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::NAKED) {
        trace!("InstrumentCoverage skipped for {def_id:?} (`#[naked]`)");
        return false;
    }

    if !tcx.coverage_attr_on(def_id) {
        trace!("InstrumentCoverage skipped for {def_id:?} (`#[coverage(off)]`)");
        return false;
    }

    true
}

/// Query implementation for `coverage_attr_on`.
fn coverage_attr_on(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    // Check for annotations directly on this def.
    if let Some(attr) = tcx.get_attr(def_id, sym::coverage) {
        match attr.meta_item_list().as_deref() {
            Some([item]) if item.has_name(sym::off) => return false,
            Some([item]) if item.has_name(sym::on) => return true,
            Some(_) | None => {
                // Other possibilities should have been rejected by `rustc_parse::validate_attr`.
                // Use `span_delayed_bug` to avoid an ICE in failing builds (#127880).
                tcx.dcx().span_delayed_bug(attr.span, "unexpected value of coverage attribute");
            }
        }
    }

    match tcx.opt_local_parent(def_id) {
        // Check the parent def (and so on recursively) until we find an
        // enclosing attribute or reach the crate root.
        Some(parent) => tcx.coverage_attr_on(parent),
        // We reached the crate root without seeing a coverage attribute, so
        // allow coverage instrumentation by default.
        None => true,
    }
}

/// Query implementation for `coverage_ids_info`.
fn coverage_ids_info<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance_def: ty::InstanceKind<'tcx>,
) -> Option<CoverageIdsInfo> {
    let mir_body = tcx.instance_mir(instance_def);
    let fn_cov_info = mir_body.function_coverage_info.as_deref()?;

    let mut counters_seen = DenseBitSet::new_empty(fn_cov_info.num_counters);
    let mut expressions_seen = DenseBitSet::new_filled(fn_cov_info.expressions.len());

    // For each expression ID that is directly used by one or more mappings,
    // mark it as not-yet-seen. This indicates that we expect to see a
    // corresponding `ExpressionUsed` statement during MIR traversal.
    for mapping in fn_cov_info.mappings.iter() {
        // Currently we only worry about ordinary code mappings.
        // For branch and MC/DC mappings, expressions might not correspond
        // to any particular point in the control-flow graph.
        // (Keep this in sync with the injection of `ExpressionUsed`
        // statements in the `InstrumentCoverage` MIR pass.)
        if let MappingKind::Code(CovTerm::Expression(id)) = mapping.kind {
            expressions_seen.remove(id);
        }
    }

    for kind in all_coverage_in_mir_body(mir_body) {
        match *kind {
            CoverageKind::CounterIncrement { id } => {
                counters_seen.insert(id);
            }
            CoverageKind::ExpressionUsed { id } => {
                expressions_seen.insert(id);
            }
            _ => {}
        }
    }

    let zero_expressions =
        identify_zero_expressions(fn_cov_info, &counters_seen, &expressions_seen);

    Some(CoverageIdsInfo { counters_seen, zero_expressions })
}

fn all_coverage_in_mir_body<'a, 'tcx>(
    body: &'a Body<'tcx>,
) -> impl Iterator<Item = &'a CoverageKind> + Captures<'tcx> {
    body.basic_blocks.iter().flat_map(|bb_data| &bb_data.statements).filter_map(|statement| {
        match statement.kind {
            StatementKind::Coverage(ref kind) if !is_inlined(body, statement) => Some(kind),
            _ => None,
        }
    })
}

fn is_inlined(body: &Body<'_>, statement: &Statement<'_>) -> bool {
    let scope_data = &body.source_scopes[statement.source_info.scope];
    scope_data.inlined.is_some() || scope_data.inlined_parent_scope.is_some()
}

/// Identify expressions that will always have a value of zero, and note their
/// IDs in a `DenseBitSet`. Mappings that refer to a zero expression can instead
/// become mappings to a constant zero value.
///
/// This function mainly exists to preserve the simplifications that were
/// already being performed by the Rust-side expression renumbering, so that
/// the resulting coverage mappings don't get worse.
fn identify_zero_expressions(
    fn_cov_info: &FunctionCoverageInfo,
    counters_seen: &DenseBitSet<CounterId>,
    expressions_seen: &DenseBitSet<ExpressionId>,
) -> DenseBitSet<ExpressionId> {
    // The set of expressions that either were optimized out entirely, or
    // have zero as both of their operands, and will therefore always have
    // a value of zero. Other expressions that refer to these as operands
    // can have those operands replaced with `CovTerm::Zero`.
    let mut zero_expressions = DenseBitSet::new_empty(fn_cov_info.expressions.len());

    // Simplify a copy of each expression based on lower-numbered expressions,
    // and then update the set of always-zero expressions if necessary.
    // (By construction, expressions can only refer to other expressions
    // that have lower IDs, so one pass is sufficient.)
    for (id, expression) in fn_cov_info.expressions.iter_enumerated() {
        if !expressions_seen.contains(id) {
            // If an expression was not seen, it must have been optimized away,
            // so any operand that refers to it can be replaced with zero.
            zero_expressions.insert(id);
            continue;
        }

        // We don't need to simplify the actual expression data in the
        // expressions list; we can just simplify a temporary copy and then
        // use that to update the set of always-zero expressions.
        let Expression { mut lhs, op, mut rhs } = *expression;

        // If an expression has an operand that is also an expression, the
        // operand's ID must be strictly lower. This is what lets us find
        // all zero expressions in one pass.
        let assert_operand_expression_is_lower = |operand_id: ExpressionId| {
            assert!(
                operand_id < id,
                "Operand {operand_id:?} should be less than {id:?} in {expression:?}",
            )
        };

        // If an operand refers to a counter or expression that is always
        // zero, then that operand can be replaced with `CovTerm::Zero`.
        let maybe_set_operand_to_zero = |operand: &mut CovTerm| {
            if let CovTerm::Expression(id) = *operand {
                assert_operand_expression_is_lower(id);
            }

            if is_zero_term(&counters_seen, &zero_expressions, *operand) {
                *operand = CovTerm::Zero;
            }
        };
        maybe_set_operand_to_zero(&mut lhs);
        maybe_set_operand_to_zero(&mut rhs);

        // Coverage counter values cannot be negative, so if an expression
        // involves subtraction from zero, assume that its RHS must also be zero.
        // (Do this after simplifications that could set the LHS to zero.)
        if lhs == CovTerm::Zero && op == Op::Subtract {
            rhs = CovTerm::Zero;
        }

        // After the above simplifications, if both operands are zero, then
        // we know that this expression is always zero too.
        if lhs == CovTerm::Zero && rhs == CovTerm::Zero {
            zero_expressions.insert(id);
        }
    }

    zero_expressions
}

/// Returns `true` if the given term is known to have a value of zero, taking
/// into account knowledge of which counters are unused and which expressions
/// are always zero.
fn is_zero_term(
    counters_seen: &DenseBitSet<CounterId>,
    zero_expressions: &DenseBitSet<ExpressionId>,
    term: CovTerm,
) -> bool {
    match term {
        CovTerm::Zero => true,
        CovTerm::Counter(id) => !counters_seen.contains(id),
        CovTerm::Expression(id) => zero_expressions.contains(id),
    }
}
