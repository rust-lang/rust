use crate::coverageinfo::ffi::{Counter, CounterExpression, ExprKind};

use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxIndexSet;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::coverage::{
    CodeRegion, CounterId, CovTerm, Expression, ExpressionId, FunctionCoverageInfo, Mapping, Op,
};
use rustc_middle::ty::Instance;
use rustc_span::Symbol;

/// Holds all of the coverage mapping data associated with a function instance,
/// collected during traversal of `Coverage` statements in the function's MIR.
#[derive(Debug)]
pub struct FunctionCoverageCollector<'tcx> {
    /// Coverage info that was attached to this function by the instrumentor.
    function_coverage_info: &'tcx FunctionCoverageInfo,
    is_used: bool,

    /// Tracks which counters have been seen, so that we can identify mappings
    /// to counters that were optimized out, and set them to zero.
    counters_seen: BitSet<CounterId>,
    /// Contains all expression IDs that have been seen in an `ExpressionUsed`
    /// coverage statement, plus all expression IDs that aren't directly used
    /// by any mappings (and therefore do not have expression-used statements).
    /// After MIR traversal is finished, we can conclude that any IDs missing
    /// from this set must have had their statements deleted by MIR opts.
    expressions_seen: BitSet<ExpressionId>,
}

impl<'tcx> FunctionCoverageCollector<'tcx> {
    /// Creates a new set of coverage data for a used (called) function.
    pub fn new(
        instance: Instance<'tcx>,
        function_coverage_info: &'tcx FunctionCoverageInfo,
    ) -> Self {
        Self::create(instance, function_coverage_info, true)
    }

    /// Creates a new set of coverage data for an unused (never called) function.
    pub fn unused(
        instance: Instance<'tcx>,
        function_coverage_info: &'tcx FunctionCoverageInfo,
    ) -> Self {
        Self::create(instance, function_coverage_info, false)
    }

    fn create(
        instance: Instance<'tcx>,
        function_coverage_info: &'tcx FunctionCoverageInfo,
        is_used: bool,
    ) -> Self {
        let num_counters = function_coverage_info.num_counters;
        let num_expressions = function_coverage_info.expressions.len();
        debug!(
            "FunctionCoverage::create(instance={instance:?}) has \
            num_counters={num_counters}, num_expressions={num_expressions}, is_used={is_used}"
        );

        // Create a filled set of expression IDs, so that expressions not
        // directly used by mappings will be treated as "seen".
        // (If they end up being unused, LLVM will delete them for us.)
        let mut expressions_seen = BitSet::new_filled(num_expressions);
        // For each expression ID that is directly used by one or more mappings,
        // mark it as not-yet-seen. This indicates that we expect to see a
        // corresponding `ExpressionUsed` statement during MIR traversal.
        for Mapping { term, .. } in &function_coverage_info.mappings {
            if let &CovTerm::Expression(id) = term {
                expressions_seen.remove(id);
            }
        }

        Self {
            function_coverage_info,
            is_used,
            counters_seen: BitSet::new_empty(num_counters),
            expressions_seen,
        }
    }

    /// Marks a counter ID as having been seen in a counter-increment statement.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn mark_counter_id_seen(&mut self, id: CounterId) {
        self.counters_seen.insert(id);
    }

    /// Marks an expression ID as having been seen in an expression-used statement.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn mark_expression_id_seen(&mut self, id: ExpressionId) {
        self.expressions_seen.insert(id);
    }

    /// Identify expressions that will always have a value of zero, and note
    /// their IDs in [`ZeroExpressions`]. Mappings that refer to a zero expression
    /// can instead become mappings to a constant zero value.
    ///
    /// This method mainly exists to preserve the simplifications that were
    /// already being performed by the Rust-side expression renumbering, so that
    /// the resulting coverage mappings don't get worse.
    fn identify_zero_expressions(&self) -> ZeroExpressions {
        // The set of expressions that either were optimized out entirely, or
        // have zero as both of their operands, and will therefore always have
        // a value of zero. Other expressions that refer to these as operands
        // can have those operands replaced with `CovTerm::Zero`.
        let mut zero_expressions = ZeroExpressions::default();

        // Simplify a copy of each expression based on lower-numbered expressions,
        // and then update the set of always-zero expressions if necessary.
        // (By construction, expressions can only refer to other expressions
        // that have lower IDs, so one pass is sufficient.)
        for (id, expression) in self.function_coverage_info.expressions.iter_enumerated() {
            if !self.expressions_seen.contains(id) {
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

                if is_zero_term(&self.counters_seen, &zero_expressions, *operand) {
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

    pub(crate) fn into_finished(self) -> FunctionCoverage<'tcx> {
        let zero_expressions = self.identify_zero_expressions();
        let FunctionCoverageCollector { function_coverage_info, is_used, counters_seen, .. } = self;

        FunctionCoverage { function_coverage_info, is_used, counters_seen, zero_expressions }
    }
}

pub(crate) struct FunctionCoverage<'tcx> {
    function_coverage_info: &'tcx FunctionCoverageInfo,
    is_used: bool,

    counters_seen: BitSet<CounterId>,
    zero_expressions: ZeroExpressions,
}

impl<'tcx> FunctionCoverage<'tcx> {
    /// Returns true for a used (called) function, and false for an unused function.
    pub(crate) fn is_used(&self) -> bool {
        self.is_used
    }

    /// Return the source hash, generated from the HIR node structure, and used to indicate whether
    /// or not the source code structure changed between different compilations.
    pub fn source_hash(&self) -> u64 {
        if self.is_used { self.function_coverage_info.function_source_hash } else { 0 }
    }

    /// Returns an iterator over all filenames used by this function's mappings.
    pub(crate) fn all_file_names(&self) -> impl Iterator<Item = Symbol> + Captures<'_> {
        self.function_coverage_info.mappings.iter().map(|mapping| mapping.code_region.file_name)
    }

    /// Convert this function's coverage expression data into a form that can be
    /// passed through FFI to LLVM.
    pub(crate) fn counter_expressions(
        &self,
    ) -> impl Iterator<Item = CounterExpression> + ExactSizeIterator + Captures<'_> {
        // We know that LLVM will optimize out any unused expressions before
        // producing the final coverage map, so there's no need to do the same
        // thing on the Rust side unless we're confident we can do much better.
        // (See `CounterExpressionsMinimizer` in `CoverageMappingWriter.cpp`.)

        self.function_coverage_info.expressions.iter().map(move |&Expression { lhs, op, rhs }| {
            CounterExpression {
                lhs: self.counter_for_term(lhs),
                kind: match op {
                    Op::Add => ExprKind::Add,
                    Op::Subtract => ExprKind::Subtract,
                },
                rhs: self.counter_for_term(rhs),
            }
        })
    }

    /// Converts this function's coverage mappings into an intermediate form
    /// that will be used by `mapgen` when preparing for FFI.
    pub(crate) fn counter_regions(
        &self,
    ) -> impl Iterator<Item = (Counter, &CodeRegion)> + ExactSizeIterator {
        self.function_coverage_info.mappings.iter().map(move |mapping| {
            let &Mapping { term, ref code_region } = mapping;
            let counter = self.counter_for_term(term);
            (counter, code_region)
        })
    }

    fn counter_for_term(&self, term: CovTerm) -> Counter {
        if is_zero_term(&self.counters_seen, &self.zero_expressions, term) {
            Counter::ZERO
        } else {
            Counter::from_term(term)
        }
    }
}

/// Set of expression IDs that are known to always evaluate to zero.
/// Any mapping or expression operand that refers to these expressions can have
/// that reference replaced with a constant zero value.
#[derive(Default)]
struct ZeroExpressions(FxIndexSet<ExpressionId>);

impl ZeroExpressions {
    fn insert(&mut self, id: ExpressionId) {
        self.0.insert(id);
    }

    fn contains(&self, id: ExpressionId) -> bool {
        self.0.contains(&id)
    }
}

/// Returns `true` if the given term is known to have a value of zero, taking
/// into account knowledge of which counters are unused and which expressions
/// are always zero.
fn is_zero_term(
    counters_seen: &BitSet<CounterId>,
    zero_expressions: &ZeroExpressions,
    term: CovTerm,
) -> bool {
    match term {
        CovTerm::Zero => true,
        CovTerm::Counter(id) => !counters_seen.contains(id),
        CovTerm::Expression(id) => zero_expressions.contains(id),
    }
}
