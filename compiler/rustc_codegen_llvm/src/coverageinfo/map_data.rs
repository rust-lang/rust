pub use super::ffi::*;

use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::bug;
use rustc_middle::mir::coverage::{
    CodeRegion, CounterId, ExpressionId, MappedExpressionIndex, Op, Operand,
};
use rustc_middle::ty::Instance;
use rustc_middle::ty::TyCtxt;

#[derive(Clone, Debug, PartialEq)]
pub struct Expression {
    lhs: Operand,
    op: Op,
    rhs: Operand,
    region: Option<CodeRegion>,
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
    counters: IndexVec<CounterId, Option<CodeRegion>>,
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

    /// Adds a code region to be counted by an injected counter intrinsic.
    pub fn add_counter(&mut self, id: CounterId, region: CodeRegion) {
        if let Some(previous_region) = self.counters[id].replace(region.clone()) {
            assert_eq!(previous_region, region, "add_counter: code region for id changed");
        }
    }

    /// Both counters and "counter expressions" (or simply, "expressions") can be operands in other
    /// expressions. These are tracked as separate variants of `Operand`, so there is no ambiguity
    /// between operands that are counter IDs and operands that are expression IDs.
    pub fn add_counter_expression(
        &mut self,
        expression_id: ExpressionId,
        lhs: Operand,
        op: Op,
        rhs: Operand,
        region: Option<CodeRegion>,
    ) {
        debug!(
            "add_counter_expression({:?}, lhs={:?}, op={:?}, rhs={:?} at {:?}",
            expression_id, lhs, op, rhs, region
        );
        debug_assert!(
            expression_id.as_usize() < self.expressions.len(),
            "expression_id {} is out of range for expressions.len() = {}
            for {:?}",
            expression_id.as_usize(),
            self.expressions.len(),
            self,
        );
        if let Some(previous_expression) = self.expressions[expression_id].replace(Expression {
            lhs,
            op,
            rhs,
            region: region.clone(),
        }) {
            assert_eq!(
                previous_expression,
                Expression { lhs, op, rhs, region },
                "add_counter_expression: expression for id changed"
            );
        }
    }

    /// Add a region that will be marked as "unreachable", with a constant "zero counter".
    pub fn add_unreachable_region(&mut self, region: CodeRegion) {
        self.unreachable_regions.push(region)
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

        let counter_regions = self.counter_regions();
        let (counter_expressions, expression_regions) = self.expressions_with_regions();
        let unreachable_regions = self.unreachable_regions();

        let counter_regions =
            counter_regions.chain(expression_regions.into_iter().chain(unreachable_regions));
        (counter_expressions, counter_regions)
    }

    fn counter_regions(&self) -> impl Iterator<Item = (Counter, &CodeRegion)> {
        self.counters.iter_enumerated().filter_map(|(index, entry)| {
            // Option::map() will return None to filter out missing counters. This may happen
            // if, for example, a MIR-instrumented counter is removed during an optimization.
            entry.as_ref().map(|region| (Counter::counter_value_reference(index), region))
        })
    }

    fn expressions_with_regions(
        &self,
    ) -> (Vec<CounterExpression>, impl Iterator<Item = (Counter, &CodeRegion)>) {
        let mut counter_expressions = Vec::with_capacity(self.expressions.len());
        let mut expression_regions = Vec::with_capacity(self.expressions.len());
        let mut new_indexes = IndexVec::from_elem_n(None, self.expressions.len());

        // This closure converts any `Expression` operand (`lhs` or `rhs` of the `Op::Add` or
        // `Op::Subtract` operation) into its native `llvm::coverage::Counter::CounterKind` type
        // and value.
        //
        // Expressions will be returned from this function in a sequential vector (array) of
        // `CounterExpression`, so the expression IDs must be mapped from their original,
        // potentially sparse set of indexes.
        //
        // An `Expression` as an operand will have already been encountered as an `Expression` with
        // operands, so its new_index will already have been generated (as a 1-up index value).
        // (If an `Expression` as an operand does not have a corresponding new_index, it was
        // probably optimized out, after the expression was injected into the MIR, so it will
        // get a `CounterKind::Zero` instead.)
        //
        // In other words, an `Expression`s at any given index can include other expressions as
        // operands, but expression operands can only come from the subset of expressions having
        // `expression_index`s lower than the referencing `Expression`. Therefore, it is
        // reasonable to look up the new index of an expression operand while the `new_indexes`
        // vector is only complete up to the current `ExpressionIndex`.
        type NewIndexes = IndexSlice<ExpressionId, Option<MappedExpressionIndex>>;
        let id_to_counter = |new_indexes: &NewIndexes, operand: Operand| match operand {
            Operand::Zero => Some(Counter::zero()),
            Operand::Counter(id) => Some(Counter::counter_value_reference(id)),
            Operand::Expression(id) => {
                self.expressions
                    .get(id)
                    .expect("expression id is out of range")
                    .as_ref()
                    // If an expression was optimized out, assume it would have produced a count
                    // of zero. This ensures that expressions dependent on optimized-out
                    // expressions are still valid.
                    .map_or(Some(Counter::zero()), |_| new_indexes[id].map(Counter::expression))
            }
        };

        for (original_index, expression) in
            self.expressions.iter_enumerated().filter_map(|(original_index, entry)| {
                // Option::map() will return None to filter out missing expressions. This may happen
                // if, for example, a MIR-instrumented expression is removed during an optimization.
                entry.as_ref().map(|expression| (original_index, expression))
            })
        {
            let optional_region = &expression.region;
            let Expression { lhs, op, rhs, .. } = *expression;

            if let Some(Some((lhs_counter, mut rhs_counter))) = id_to_counter(&new_indexes, lhs)
                .map(|lhs_counter| {
                    id_to_counter(&new_indexes, rhs).map(|rhs_counter| (lhs_counter, rhs_counter))
                })
            {
                if lhs_counter.is_zero() && op.is_subtract() {
                    // The left side of a subtraction was probably optimized out. As an example,
                    // a branch condition might be evaluated as a constant expression, and the
                    // branch could be removed, dropping unused counters in the process.
                    //
                    // Since counters are unsigned, we must assume the result of the expression
                    // can be no more and no less than zero. An expression known to evaluate to zero
                    // does not need to be added to the coverage map.
                    //
                    // Coverage test `loops_branches.rs` includes multiple variations of branches
                    // based on constant conditional (literal `true` or `false`), and demonstrates
                    // that the expected counts are still correct.
                    debug!(
                        "Expression subtracts from zero (assume unreachable): \
                        original_index={:?}, lhs={:?}, op={:?}, rhs={:?}, region={:?}",
                        original_index, lhs, op, rhs, optional_region,
                    );
                    rhs_counter = Counter::zero();
                }
                debug_assert!(
                    lhs_counter.is_zero()
                        // Note: with `as usize` the ID _could_ overflow/wrap if `usize = u16`
                        || ((lhs_counter.zero_based_id() as usize)
                            <= usize::max(self.counters.len(), self.expressions.len())),
                    "lhs id={} > both counters.len()={} and expressions.len()={}
                    ({:?} {:?} {:?})",
                    lhs_counter.zero_based_id(),
                    self.counters.len(),
                    self.expressions.len(),
                    lhs_counter,
                    op,
                    rhs_counter,
                );

                debug_assert!(
                    rhs_counter.is_zero()
                        // Note: with `as usize` the ID _could_ overflow/wrap if `usize = u16`
                        || ((rhs_counter.zero_based_id() as usize)
                            <= usize::max(self.counters.len(), self.expressions.len())),
                    "rhs id={} > both counters.len()={} and expressions.len()={}
                    ({:?} {:?} {:?})",
                    rhs_counter.zero_based_id(),
                    self.counters.len(),
                    self.expressions.len(),
                    lhs_counter,
                    op,
                    rhs_counter,
                );

                // Both operands exist. `Expression` operands exist in `self.expressions` and have
                // been assigned a `new_index`.
                let mapped_expression_index =
                    MappedExpressionIndex::from(counter_expressions.len());
                let expression = CounterExpression::new(
                    lhs_counter,
                    match op {
                        Op::Add => ExprKind::Add,
                        Op::Subtract => ExprKind::Subtract,
                    },
                    rhs_counter,
                );
                debug!(
                    "Adding expression {:?} = {:?}, region: {:?}",
                    mapped_expression_index, expression, optional_region
                );
                counter_expressions.push(expression);
                new_indexes[original_index] = Some(mapped_expression_index);
                if let Some(region) = optional_region {
                    expression_regions.push((Counter::expression(mapped_expression_index), region));
                }
            } else {
                bug!(
                    "expression has one or more missing operands \
                      original_index={:?}, lhs={:?}, op={:?}, rhs={:?}, region={:?}",
                    original_index,
                    lhs,
                    op,
                    rhs,
                    optional_region,
                );
            }
        }
        (counter_expressions, expression_regions.into_iter())
    }

    fn unreachable_regions(&self) -> impl Iterator<Item = (Counter, &CodeRegion)> {
        self.unreachable_regions.iter().map(|region| (Counter::zero(), region))
    }
}
