use crate::coverageinfo::ffi::{Counter, CounterExpression, ExprKind};

use rustc_data_structures::fx::FxIndexSet;
use rustc_index::IndexVec;
use rustc_middle::mir::coverage::{CodeRegion, CounterId, ExpressionId, Op, Operand};
use rustc_middle::ty::Instance;
use rustc_middle::ty::TyCtxt;

#[derive(Clone, Debug, PartialEq)]
pub struct Expression {
    lhs: Operand,
    op: Op,
    rhs: Operand,
    code_regions: Vec<CodeRegion>,
}

/// Collects all of the coverage regions associated with (a) injected counters, (b) counter
/// expressions (additions or subtraction), and (c) unreachable regions (always counted as zero),
/// for a given Function. This struct also stores the `function_source_hash`,
/// computed during instrumentation, and forwarded with counters.
///
/// Note, it may be important to understand LLVM's definitions of `unreachable` regions versus "gap
/// regions" (or "gap areas"). A gap region is a code region within a counted region (either counter
/// or expression), but the line or lines in the gap region are not executable (such as lines with
/// only whitespace or comments). According to LLVM Code Coverage Mapping documentation, "A count
/// for a gap area is only used as the line execution count if there are no other regions on a
/// line."
#[derive(Debug)]
pub struct FunctionCoverage<'tcx> {
    instance: Instance<'tcx>,
    source_hash: u64,
    is_used: bool,
    counters: IndexVec<CounterId, Option<Vec<CodeRegion>>>,
    expressions: IndexVec<ExpressionId, Option<Expression>>,
    unreachable_regions: Vec<CodeRegion>,
}

impl<'tcx> FunctionCoverage<'tcx> {
    /// Creates a new set of coverage data for a used (called) function.
    pub fn new(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Self {
        Self::create(tcx, instance, true)
    }

    /// Creates a new set of coverage data for an unused (never called) function.
    pub fn unused(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Self {
        Self::create(tcx, instance, false)
    }

    fn create(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>, is_used: bool) -> Self {
        let coverageinfo = tcx.coverageinfo(instance.def);
        debug!(
            "FunctionCoverage::create(instance={:?}) has coverageinfo={:?}. is_used={}",
            instance, coverageinfo, is_used
        );
        Self {
            instance,
            source_hash: 0, // will be set with the first `add_counter()`
            is_used,
            counters: IndexVec::from_elem_n(None, coverageinfo.num_counters as usize),
            expressions: IndexVec::from_elem_n(None, coverageinfo.num_expressions as usize),
            unreachable_regions: Vec::new(),
        }
    }

    /// Returns true for a used (called) function, and false for an unused function.
    pub fn is_used(&self) -> bool {
        self.is_used
    }

    /// Sets the function source hash value. If called multiple times for the same function, all
    /// calls should have the same hash value.
    pub fn set_function_source_hash(&mut self, source_hash: u64) {
        if self.source_hash == 0 {
            self.source_hash = source_hash;
        } else {
            debug_assert_eq!(source_hash, self.source_hash);
        }
    }

    /// Adds code regions to be counted by an injected counter intrinsic.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn add_counter(&mut self, id: CounterId, code_regions: &[CodeRegion]) {
        if code_regions.is_empty() {
            return;
        }

        let slot = &mut self.counters[id];
        match slot {
            None => *slot = Some(code_regions.to_owned()),
            // If this counter ID slot has already been filled, it should
            // contain identical information.
            Some(ref previous_regions) => assert_eq!(
                previous_regions, code_regions,
                "add_counter: code regions for id changed"
            ),
        }
    }

    /// Adds information about a coverage expression, along with zero or more
    /// code regions mapped to that expression.
    ///
    /// Both counters and "counter expressions" (or simply, "expressions") can be operands in other
    /// expressions. These are tracked as separate variants of `Operand`, so there is no ambiguity
    /// between operands that are counter IDs and operands that are expression IDs.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn add_counter_expression(
        &mut self,
        expression_id: ExpressionId,
        lhs: Operand,
        op: Op,
        rhs: Operand,
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

        let expression = Expression { lhs, op, rhs, code_regions: code_regions.to_owned() };
        let slot = &mut self.expressions[expression_id];
        match slot {
            None => *slot = Some(expression),
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
        self.unreachable_regions.extend_from_slice(code_regions);
    }

    /// Perform some simplifications to make the final coverage mappings
    /// slightly smaller.
    ///
    /// This method mainly exists to preserve the simplifications that were
    /// already being performed by the Rust-side expression renumbering, so that
    /// the resulting coverage mappings don't get worse.
    pub(crate) fn simplify_expressions(&mut self) {
        // The set of expressions that either were optimized out entirely, or
        // have zero as both of their operands, and will therefore always have
        // a value of zero. Other expressions that refer to these as operands
        // can have those operands replaced with `Operand::Zero`.
        let mut zero_expressions = FxIndexSet::default();

        // For each expression, perform simplifications based on lower-numbered
        // expressions, and then update the set of always-zero expressions if
        // necessary.
        // (By construction, expressions can only refer to other expressions
        // that have lower IDs, so one simplification pass is sufficient.)
        for (id, maybe_expression) in self.expressions.iter_enumerated_mut() {
            let Some(expression) = maybe_expression else {
                // If an expression is missing, it must have been optimized away,
                // so any operand that refers to it can be replaced with zero.
                zero_expressions.insert(id);
                continue;
            };

            // If an operand refers to an expression that is always zero, then
            // that operand can be replaced with `Operand::Zero`.
            let maybe_set_operand_to_zero = |operand: &mut Operand| match &*operand {
                Operand::Expression(id) if zero_expressions.contains(id) => {
                    *operand = Operand::Zero;
                }
                _ => (),
            };
            maybe_set_operand_to_zero(&mut expression.lhs);
            maybe_set_operand_to_zero(&mut expression.rhs);

            // Coverage counter values cannot be negative, so if an expression
            // involves subtraction from zero, assume that its RHS must also be zero.
            // (Do this after simplifications that could set the LHS to zero.)
            if let Expression { lhs: Operand::Zero, op: Op::Subtract, .. } = expression {
                expression.rhs = Operand::Zero;
            }

            // After the above simplifications, if both operands are zero, then
            // we know that this expression is always zero too.
            if let Expression { lhs: Operand::Zero, rhs: Operand::Zero, .. } = expression {
                zero_expressions.insert(id);
            }
        }
    }

    /// Return the source hash, generated from the HIR node structure, and used to indicate whether
    /// or not the source code structure changed between different compilations.
    pub fn source_hash(&self) -> u64 {
        self.source_hash
    }

    /// Generate an array of CounterExpressions, and an iterator over all `Counter`s and their
    /// associated `Regions` (from which the LLVM-specific `CoverageMapGenerator` will create
    /// `CounterMappingRegion`s.
    pub fn get_expressions_and_counter_regions(
        &self,
    ) -> (Vec<CounterExpression>, impl Iterator<Item = (Counter, &CodeRegion)>) {
        assert!(
            self.source_hash != 0 || !self.is_used,
            "No counters provided the source_hash for used function: {:?}",
            self.instance
        );

        let counter_expressions = self.counter_expressions();
        // Expression IDs are indices into `self.expressions`, and on the LLVM
        // side they will be treated as indices into `counter_expressions`, so
        // the two vectors should correspond 1:1.
        assert_eq!(self.expressions.len(), counter_expressions.len());

        let counter_regions = self.counter_regions();
        let expression_regions = self.expression_regions();
        let unreachable_regions = self.unreachable_regions();

        let counter_regions =
            counter_regions.chain(expression_regions.into_iter().chain(unreachable_regions));
        (counter_expressions, counter_regions)
    }

    fn counter_regions(&self) -> impl Iterator<Item = (Counter, &CodeRegion)> {
        self.counters
            .iter_enumerated()
            // Filter out counter IDs that we never saw during MIR traversal.
            // This can happen if a counter was optimized out by MIR transforms
            // (and replaced with `CoverageKind::Unreachable` instead).
            .filter_map(|(id, maybe_code_regions)| Some((id, maybe_code_regions.as_ref()?)))
            .flat_map(|(id, code_regions)| {
                let counter = Counter::counter_value_reference(id);
                code_regions.iter().map(move |region| (counter, region))
            })
    }

    /// Convert this function's coverage expression data into a form that can be
    /// passed through FFI to LLVM.
    fn counter_expressions(&self) -> Vec<CounterExpression> {
        // We know that LLVM will optimize out any unused expressions before
        // producing the final coverage map, so there's no need to do the same
        // thing on the Rust side unless we're confident we can do much better.
        // (See `CounterExpressionsMinimizer` in `CoverageMappingWriter.cpp`.)

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
                        Counter::from_operand(lhs),
                        match op {
                            Op::Add => ExprKind::Add,
                            Op::Subtract => ExprKind::Subtract,
                        },
                        Counter::from_operand(rhs),
                    )
                }
            })
            .collect::<Vec<_>>()
    }

    fn expression_regions(&self) -> Vec<(Counter, &CodeRegion)> {
        // Find all of the expression IDs that weren't optimized out AND have
        // one or more attached code regions, and return the corresponding
        // mappings as counter/region pairs.
        self.expressions
            .iter_enumerated()
            .filter_map(|(id, maybe_expression)| {
                let code_regions = &maybe_expression.as_ref()?.code_regions;
                Some((id, code_regions))
            })
            .flat_map(|(id, code_regions)| {
                let counter = Counter::expression(id);
                code_regions.iter().map(move |code_region| (counter, code_region))
            })
            .collect::<Vec<_>>()
    }

    fn unreachable_regions(&self) -> impl Iterator<Item = (Counter, &CodeRegion)> {
        self.unreachable_regions.iter().map(|region| (Counter::ZERO, region))
    }
}
