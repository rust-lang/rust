use crate::coverageinfo::ffi::{Counter, CounterExpression, ExprKind};

use rustc_data_structures::fx::FxHashMap;
use rustc_index::bit_set::BitSet;
use rustc_index::IndexVec;
use rustc_middle::mir::coverage::{CounterId, ExpressionId, Op, Operand};
use rustc_middle::ty::Instance;
use rustc_middle::ty::TyCtxt;

#[derive(Clone, Debug, PartialEq)]
pub struct Expression {
    lhs: Operand,
    op: Op,
    rhs: Operand,
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
pub struct FunctionCoverage {
    source_hash: u64,
    is_used: bool,
    counters: BitSet<CounterId>,
    expressions: IndexVec<ExpressionId, Option<Expression>>,
}

impl<'tcx> FunctionCoverage {
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
            source_hash: 0, // will be set with the first `add_counter()`
            is_used,
            counters: BitSet::new_empty(coverageinfo.num_counters as usize),
            expressions: IndexVec::from_elem_n(None, coverageinfo.num_expressions as usize),
        }
    }

    /// Returns true for a used (called) function, and false for an unused function.
    pub fn is_used(&self) -> bool {
        self.is_used
    }

    /// Adds a Counter, along with setting the function source hash value.
    /// If called multiple times for the same function,
    /// all calls should have the same `source_hash` value.
    pub fn add_counter(&mut self, source_hash: u64, id: CounterId) {
        if self.source_hash == 0 {
            self.source_hash = source_hash;
        } else {
            debug_assert_eq!(source_hash, self.source_hash);
        }

        self.counters.insert(id);
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
    ) {
        debug!(
            "add_counter_expression({:?}, lhs={:?}, op={:?}, rhs={:?}",
            expression_id, lhs, op, rhs
        );
        debug_assert!(
            expression_id.as_usize() < self.expressions.len(),
            "expression_id {} is out of range for expressions.len() = {}
            for {:?}",
            expression_id.as_usize(),
            self.expressions.len(),
            self,
        );
        if let Some(previous_expression) =
            self.expressions[expression_id].replace(Expression { lhs, op, rhs })
        {
            assert_eq!(
                previous_expression,
                Expression { lhs, op, rhs },
                "add_counter_expression: expression for id changed"
            );
        }
    }

    /// Perform some simplifications to make the final coverage mappings
    /// slightly smaller.
    pub(crate) fn simplify_expressions(&mut self) {
        // The set of expressions that were simplified to either `Zero` or a
        // `Counter`. Other expressions that refer to these as operands
        // can then also be simplified.
        let mut simplified_expressions = FxHashMap::default();

        // For each expression, perform simplifications based on lower-numbered
        // expressions, and then update the map of simplified expressions if
        // necessary.
        // (By construction, expressions can only refer to other expressions
        // that have lower IDs, so one simplification pass is sufficient.)
        for (id, maybe_expression) in self.expressions.iter_enumerated_mut() {
            let Some(expression) = maybe_expression else {
                // If an expression is missing, it must have been optimized away,
                // so any operand that refers to it can be replaced with zero.
                simplified_expressions.insert(id, Operand::Zero);
                continue;
            };

            // If an operand refers to an expression that has been simplified, then
            // replace that operand with the simplified version.
            let maybe_simplify_operand = |operand: &mut Operand| match operand {
                Operand::Zero => {}
                Operand::Counter(id) => {
                    if !self.counters.contains(*id) {
                        *operand = Operand::Zero;
                    }
                }
                Operand::Expression(id) => {
                    if let Some(simplified) = simplified_expressions.get(id) {
                        *operand = *simplified;
                    }
                }
            };

            maybe_simplify_operand(&mut expression.lhs);
            maybe_simplify_operand(&mut expression.rhs);

            // Coverage counter values cannot be negative, so if an expression
            // involves subtraction from zero, assume that its RHS must also be zero.
            // (Do this after simplifications that could set the LHS to zero.)
            if let Expression { lhs: Operand::Zero, op: Op::Subtract, .. } = expression {
                expression.rhs = Operand::Zero;
            }

            // After the above simplifications, if the right hand operand is zero,
            // we can replace the expression by its left hand side.
            if let Expression { lhs, rhs: Operand::Zero, .. } = expression {
                simplified_expressions.insert(id, *lhs);
            } else
            // And the same thing for the left hand side.
            if let Expression { lhs: Operand::Zero, rhs, .. } = expression {
                simplified_expressions.insert(id, *rhs);
            }
        }
    }

    /// This will further simplify any expression, "inlining" the left hand side operand
    /// if the right hand side is `Zero`. This is similar to `simplify_expressions` above,
    /// but works for an already referenced expression.
    pub fn simplified_operand(&self, operand: Operand) -> Counter {
        let operand = match operand {
            Operand::Zero => operand,
            Operand::Counter(id) => {
                if !self.counters.contains(id) {
                    Operand::Zero
                } else {
                    operand
                }
            }
            Operand::Expression(id) => {
                if let Some(expr) = &self.expressions[id] {
                    if expr.rhs == Operand::Zero {
                        expr.lhs
                    } else if expr.lhs == Operand::Zero {
                        expr.rhs
                    } else {
                        operand
                    }
                } else {
                    operand
                }
            }
        };

        Counter::from_operand(operand)
    }
    /// Return the source hash, generated from the HIR node structure, and used to indicate whether
    /// or not the source code structure changed between different compilations.
    pub fn source_hash(&self) -> u64 {
        self.source_hash
    }

    /// Convert this function's coverage expression data into a form that can be
    /// passed through FFI to LLVM.
    pub fn counter_expressions(&self) -> Vec<CounterExpression> {
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
                &Some(Expression { lhs, op, rhs, .. }) => CounterExpression::new(
                    Counter::from_operand(lhs),
                    match op {
                        Op::Add => ExprKind::Add,
                        Op::Subtract => ExprKind::Subtract,
                    },
                    Counter::from_operand(rhs),
                ),
            })
            .collect::<Vec<_>>()
    }
}
