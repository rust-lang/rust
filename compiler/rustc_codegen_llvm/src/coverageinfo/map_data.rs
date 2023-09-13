use crate::coverageinfo::ffi::{Counter, CounterExpression, ExprKind};

use rustc_data_structures::fx::FxIndexSet;
use rustc_index::bit_set::BitSet;
use rustc_index::IndexVec;
use rustc_middle::mir::coverage::{
    CodeRegion, CounterId, CovTerm, ExpressionId, FunctionCoverageInfo, Mapping, Op,
};
use rustc_middle::ty::Instance;

#[derive(Clone, Debug, PartialEq)]
pub struct Expression {
    lhs: CovTerm,
    op: Op,
    rhs: CovTerm,
}

/// Holds all of the coverage mapping data associated with a function instance,
/// collected during traversal of `Coverage` statements in the function's MIR.
#[derive(Debug)]
pub struct FunctionCoverage<'tcx> {
    /// Coverage info that was attached to this function by the instrumentor.
    function_coverage_info: &'tcx FunctionCoverageInfo,
    is_used: bool,

    /// Tracks which counters have been seen, to avoid duplicate mappings
    /// that might be introduced by MIR inlining.
    counters_seen: BitSet<CounterId>,
    expressions: IndexVec<ExpressionId, Option<Expression>>,
    mappings: Vec<Mapping>,
}

impl<'tcx> FunctionCoverage<'tcx> {
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
        let num_expressions = function_coverage_info.num_expressions;
        debug!(
            "FunctionCoverage::create(instance={instance:?}) has \
            num_counters={num_counters}, num_expressions={num_expressions}, is_used={is_used}"
        );
        Self {
            function_coverage_info,
            is_used,
            counters_seen: BitSet::new_empty(num_counters),
            expressions: IndexVec::from_elem_n(None, num_expressions),
            mappings: Vec::new(),
        }
    }

    /// Returns true for a used (called) function, and false for an unused function.
    pub fn is_used(&self) -> bool {
        self.is_used
    }

    /// Adds code regions to be counted by an injected counter intrinsic.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn add_counter(&mut self, id: CounterId, code_regions: &[CodeRegion]) {
        if self.counters_seen.insert(id) {
            self.add_mappings(CovTerm::Counter(id), code_regions);
        }
    }

    /// Adds information about a coverage expression, along with zero or more
    /// code regions mapped to that expression.
    ///
    /// Both counters and "counter expressions" (or simply, "expressions") can be operands in other
    /// expressions. These are tracked as separate variants of `CovTerm`, so there is no ambiguity
    /// between operands that are counter IDs and operands that are expression IDs.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn add_counter_expression(
        &mut self,
        expression_id: ExpressionId,
        lhs: CovTerm,
        op: Op,
        rhs: CovTerm,
        code_regions: &[CodeRegion],
    ) {
        debug_assert!(
            expression_id.as_usize() < self.expressions.len(),
            "expression_id {} is out of range for expressions.len() = {}
            for {:?}",
            expression_id.as_usize(),
            self.expressions.len(),
            self,
        );

        let expression = Expression { lhs, op, rhs };
        let slot = &mut self.expressions[expression_id];
        match slot {
            None => {
                *slot = Some(expression);
                self.add_mappings(CovTerm::Expression(expression_id), code_regions);
            }
            // If this expression ID slot has already been filled, it should
            // contain identical information.
            Some(ref previous_expression) => assert_eq!(
                previous_expression, &expression,
                "add_counter_expression: expression for id changed"
            ),
        }
    }

    /// Adds regions that will be marked as "unreachable", with a constant "zero counter".
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn add_unreachable_regions(&mut self, code_regions: &[CodeRegion]) {
        assert!(!code_regions.is_empty(), "unreachable regions always have code regions");
        self.add_mappings(CovTerm::Zero, code_regions);
    }

    #[instrument(level = "debug", skip(self))]
    fn add_mappings(&mut self, term: CovTerm, code_regions: &[CodeRegion]) {
        self.mappings
            .extend(code_regions.iter().cloned().map(|code_region| Mapping { term, code_region }));
    }

    pub(crate) fn finalize(&mut self) {
        // Reorder the collected mappings so that counter mappings are first and
        // zero mappings are last, matching the historical order.
        self.mappings.sort_by_key(|mapping| match mapping.term {
            CovTerm::Counter(_) => 0,
            CovTerm::Expression(_) => 1,
            CovTerm::Zero => u8::MAX,
        });
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
        let mut zero_expressions = FxIndexSet::default();

        // Simplify a copy of each expression based on lower-numbered expressions,
        // and then update the set of always-zero expressions if necessary.
        // (By construction, expressions can only refer to other expressions
        // that have lower IDs, so one pass is sufficient.)
        for (id, maybe_expression) in self.expressions.iter_enumerated() {
            let Some(expression) = maybe_expression else {
                // If an expression is missing, it must have been optimized away,
                // so any operand that refers to it can be replaced with zero.
                zero_expressions.insert(id);
                continue;
            };

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

            // If an operand refers to an expression that is always zero, then
            // that operand can be replaced with `CovTerm::Zero`.
            let maybe_set_operand_to_zero = |operand: &mut CovTerm| match *operand {
                CovTerm::Expression(id) => {
                    assert_operand_expression_is_lower(id);
                    if zero_expressions.contains(&id) {
                        *operand = CovTerm::Zero;
                    }
                }
                _ => (),
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

        ZeroExpressions(zero_expressions)
    }

    /// Return the source hash, generated from the HIR node structure, and used to indicate whether
    /// or not the source code structure changed between different compilations.
    pub fn source_hash(&self) -> u64 {
        if self.is_used { self.function_coverage_info.function_source_hash } else { 0 }
    }

    /// Generate an array of CounterExpressions, and an iterator over all `Counter`s and their
    /// associated `Regions` (from which the LLVM-specific `CoverageMapGenerator` will create
    /// `CounterMappingRegion`s.
    pub fn get_expressions_and_counter_regions(
        &self,
    ) -> (Vec<CounterExpression>, impl Iterator<Item = (Counter, &CodeRegion)>) {
        let zero_expressions = self.identify_zero_expressions();

        let counter_expressions = self.counter_expressions(&zero_expressions);
        // Expression IDs are indices into `self.expressions`, and on the LLVM
        // side they will be treated as indices into `counter_expressions`, so
        // the two vectors should correspond 1:1.
        assert_eq!(self.expressions.len(), counter_expressions.len());

        let counter_regions = self.counter_regions();

        (counter_expressions, counter_regions)
    }

    /// Convert this function's coverage expression data into a form that can be
    /// passed through FFI to LLVM.
    fn counter_expressions(&self, zero_expressions: &ZeroExpressions) -> Vec<CounterExpression> {
        // We know that LLVM will optimize out any unused expressions before
        // producing the final coverage map, so there's no need to do the same
        // thing on the Rust side unless we're confident we can do much better.
        // (See `CounterExpressionsMinimizer` in `CoverageMappingWriter.cpp`.)

        let counter_from_operand = |operand: CovTerm| match operand {
            CovTerm::Expression(id) if zero_expressions.contains(id) => Counter::ZERO,
            _ => Counter::from_term(operand),
        };

        self.expressions
            .iter()
            .map(|expression| match expression {
                None => {
                    // This expression ID was allocated, but we never saw the
                    // actual expression, so it must have been optimized out.
                    // Replace it with a dummy expression, and let LLVM take
                    // care of omitting it from the expression list.
                    CounterExpression::DUMMY
                }
                &Some(Expression { lhs, op, rhs, .. }) => {
                    // Convert the operands and operator as normal.
                    CounterExpression::new(
                        counter_from_operand(lhs),
                        match op {
                            Op::Add => ExprKind::Add,
                            Op::Subtract => ExprKind::Subtract,
                        },
                        counter_from_operand(rhs),
                    )
                }
            })
            .collect::<Vec<_>>()
    }

    /// Converts this function's coverage mappings into an intermediate form
    /// that will be used by `mapgen` when preparing for FFI.
    fn counter_regions(&self) -> impl Iterator<Item = (Counter, &CodeRegion)> {
        self.mappings.iter().map(|&Mapping { term, ref code_region }| {
            let counter = Counter::from_term(term);
            (counter, code_region)
        })
    }
}

/// Set of expression IDs that are known to always evaluate to zero.
/// Any mapping or expression operand that refers to these expressions can have
/// that reference replaced with a constant zero value.
struct ZeroExpressions(FxIndexSet<ExpressionId>);

impl ZeroExpressions {
    fn contains(&self, id: ExpressionId) -> bool {
        self.0.contains(&id)
    }
}
