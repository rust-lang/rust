//! A pass that propagates the known ranges of SSA locals.
//! We can know the ranges of SSA locals in certain locations for the following code:
//! ```
//! fn foo(a: u32) {
//!   let b = a < 9; // the integer representation of b is within the full range [0, 2).
//!   if b {
//!     let c = b; // c is true since b is within the range [1, 2).
//!     let d = a; // a is within the range [0, 9), but is not a known constant.
//!   }
//! }
//! ```
//! Dominating comparison facts are intersected to discover singleton ranges.
//! Non-singleton ranges are not yet used to fold comparisons directly.
use rustc_abi::WrappingRange;
use rustc_const_eval::interpret::Scalar;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::{BasicBlock, Body, Location, Operand, Place, TerminatorKind, *};
use rustc_middle::ty::{TyCtxt, TypingEnv};
use rustc_span::DUMMY_SP;

use crate::PassPolicy;
use crate::ssa::SsaLocals;

pub(super) struct SsaRangePropagation;

impl<'tcx> crate::MirPass<'tcx> for SsaRangePropagation {
    fn policy(&self, ctx: &crate::PassCtx<'_>) -> PassPolicy {
        PassPolicy::optional(ctx.mir_opt_level() >= 2)
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let typing_env = body.typing_env(tcx);
        let ssa = SsaLocals::new(tcx, body, typing_env);
        // Clone dominators because we need them while mutating the body.
        let dominators = body.basic_blocks.dominators().clone();
        let mut range_set =
            RangeSet::new(tcx, typing_env, body, &ssa, &body.local_decls, dominators);

        let reverse_postorder = body.basic_blocks.reverse_postorder().to_vec();
        for bb in reverse_postorder {
            let data = &mut body.basic_blocks.as_mut_preserves_cfg()[bb];
            range_set.visit_basic_block_data(bb, data);
        }
    }
}

#[derive(Clone, Copy)]
struct Comparison<'tcx> {
    op: BinOp,
    input: Place<'tcx>,
    value: u128,
    max: u128,
}

impl<'tcx> Comparison<'tcx> {
    fn range(self, expected: bool) -> Option<WrappingRange> {
        let (start, end) = match (self.op, expected) {
            (BinOp::Eq, true) | (BinOp::Ne, false) => (self.value, self.value),
            (BinOp::Lt, true) => (0, self.value.checked_sub(1)?),
            (BinOp::Lt, false) => (self.value, self.max),
            (BinOp::Le, true) => (0, self.value),
            (BinOp::Le, false) => (self.value.checked_add(1)?, self.max),
            (BinOp::Gt, true) => (self.value.checked_add(1)?, self.max),
            (BinOp::Gt, false) => (0, self.value),
            (BinOp::Ge, true) => (self.value, self.max),
            (BinOp::Ge, false) => (0, self.value.checked_sub(1)?),
            _ => return None,
        };

        (start <= end).then_some(WrappingRange { start, end })
    }
}

struct RangeSet<'tcx, 'body, 'a> {
    tcx: TyCtxt<'tcx>,
    typing_env: TypingEnv<'tcx>,
    ssa: &'a SsaLocals,
    local_decls: &'body LocalDecls<'tcx>,
    dominators: Dominators<BasicBlock>,
    /// Known ranges at each locations.
    ranges: FxHashMap<Place<'tcx>, Vec<(Location, WrappingRange)>>,
    /// Comparisons whose boolean result is an SSA place.
    comparisons: FxHashMap<Place<'tcx>, Comparison<'tcx>>,
    /// Determines if the basic block has a single unique predecessor.
    unique_predecessors: DenseBitSet<BasicBlock>,
}

impl<'tcx, 'body, 'a> RangeSet<'tcx, 'body, 'a> {
    fn new(
        tcx: TyCtxt<'tcx>,
        typing_env: TypingEnv<'tcx>,
        body: &Body<'tcx>,
        ssa: &'a SsaLocals,
        local_decls: &'body LocalDecls<'tcx>,
        dominators: Dominators<BasicBlock>,
    ) -> Self {
        let predecessors = body.basic_blocks.predecessors();
        let mut unique_predecessors = DenseBitSet::new_empty(body.basic_blocks.len());
        for bb in body.basic_blocks.indices() {
            if predecessors[bb].len() == 1 {
                unique_predecessors.insert(bb);
            }
        }
        RangeSet {
            tcx,
            typing_env,
            ssa,
            local_decls,
            dominators,
            ranges: FxHashMap::default(),
            comparisons: FxHashMap::default(),
            unique_predecessors,
        }
    }

    /// Create a new known range at the location.
    fn insert_range(&mut self, place: Place<'tcx>, location: Location, range: WrappingRange) {
        assert!(self.is_ssa(place));
        self.ranges.entry(place).or_default().push((location, range));
    }

    fn insert_comparison_range(
        &mut self,
        condition: Place<'tcx>,
        location: Location,
        expected: bool,
    ) {
        let Some(comparison) = self.comparisons.get(&condition).copied() else {
            return;
        };
        let Some(range) = comparison.range(expected) else {
            return;
        };
        self.insert_range(comparison.input, location, range);
    }

    fn comparison_from_rvalue(&self, rvalue: &Rvalue<'tcx>) -> Option<Comparison<'tcx>> {
        let Rvalue::BinaryOp(op, operands) = rvalue else {
            return None;
        };
        let (lhs, rhs) = &**operands;
        if !matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne) {
            return None;
        }

        let (op, lhs, rhs) = match (lhs, rhs) {
            (_, Operand::Constant(rhs)) => (*op, lhs, rhs),
            (Operand::Constant(lhs), rhs) => {
                let op = match op {
                    BinOp::Lt => BinOp::Gt,
                    BinOp::Le => BinOp::Ge,
                    BinOp::Gt => BinOp::Lt,
                    BinOp::Ge => BinOp::Le,
                    BinOp::Eq | BinOp::Ne => *op,
                    _ => return None,
                };
                (op, rhs, lhs)
            }
            _ => return None,
        };

        let input = lhs.place()?;
        if !self.is_ssa(input) {
            return None;
        }

        let ty = lhs.ty(self.local_decls, self.tcx);
        if !ty.is_integral() || ty.is_signed() {
            return None;
        }

        let layout = self.tcx.layout_of(self.typing_env.as_query_input(ty)).ok()?;
        if rhs.const_.ty() != ty {
            return None;
        }
        let scalar = rhs.const_.try_eval_scalar_int(self.tcx, self.typing_env)?;
        // Check the size explicitly: ScalarInt::to_bits panics on a mismatch.
        if scalar.size() != layout.size {
            return None;
        }
        let value = scalar.to_bits(layout.size);

        Some(Comparison { op, input, value, max: layout.size.unsigned_int_max() })
    }

    /// Get the known range at the location.
    fn get_range(&self, place: &Place<'tcx>, location: Location) -> Option<WrappingRange> {
        let ranges = self.ranges.get(place)?;
        let mut result: Option<WrappingRange> = None;

        for (_, range) in
            ranges.iter().filter(|(range_loc, _)| range_loc.dominates(location, &self.dominators))
        {
            result = Some(match result {
                None => *range,
                Some(current) => {
                    // This pass currently creates singleton ranges and non-wrapping unsigned
                    // ranges. Bail out rather than incorrectly intersect a wrapping range.
                    if current.start > current.end || range.start > range.end {
                        return None;
                    }

                    let start = current.start.max(range.start);
                    let end = current.end.min(range.end);
                    if start > end {
                        return None;
                    }
                    WrappingRange { start, end }
                }
            });
        }

        result
    }

    fn try_as_constant(
        &mut self,
        place: Place<'tcx>,
        location: Location,
    ) -> Option<ConstOperand<'tcx>> {
        if let Some(range) = self.get_range(&place, location)
            && range.start == range.end
        {
            let ty = place.ty(self.local_decls, self.tcx).ty;
            let layout = self.tcx.layout_of(self.typing_env.as_query_input(ty)).ok()?;
            let value = ConstValue::Scalar(Scalar::from_uint(range.start, layout.size));
            let const_ = Const::Val(value, ty);
            return Some(ConstOperand { span: DUMMY_SP, user_ty: None, const_ });
        }
        None
    }

    fn is_ssa(&self, place: Place<'tcx>) -> bool {
        self.ssa.is_ssa(place.local) && place.is_stable_offset()
    }
}

impl<'tcx> MutVisitor<'tcx> for RangeSet<'tcx, '_, '_> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        // Attempts to simplify an operand to a constant value.
        if let Some(place) = operand.place()
            && let Some(const_) = self.try_as_constant(place, location)
        {
            *operand = Operand::Constant(Box::new(const_));
        };
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        // Fold known singleton operands first so they can serve as comparison constants.
        self.super_statement(statement, location);
        if let StatementKind::Assign(assign) = &statement.kind {
            let (destination, rvalue) = &**assign;
            if self.is_ssa(*destination)
                && let Some(comparison) = self.comparison_from_rvalue(rvalue)
            {
                self.comparisons.insert(*destination, comparison);
            }
        }

        match &statement.kind {
            StatementKind::Intrinsic(NonDivergingIntrinsic::Assume(operand))
                if let Some(place) = operand.place()
                    && self.is_ssa(place) =>
            {
                let successor = location.successor_within_block();
                let range = WrappingRange { start: 1, end: 1 };
                self.insert_range(place, successor, range);
                self.insert_comparison_range(place, successor, true);
            }
            _ => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &mut Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);
        match &terminator.kind {
            TerminatorKind::Assert { cond, expected, target, .. }
                if let Some(place) = cond.place()
                    && self.is_ssa(place) =>
            {
                let successor = Location { block: *target, statement_index: 0 };
                if location.strictly_dominates(successor, &self.dominators) {
                    let val = *expected as u128;
                    let range = WrappingRange { start: val, end: val };
                    self.insert_range(place, successor, range);
                    self.insert_comparison_range(place, successor, *expected);
                }
            }
            TerminatorKind::SwitchInt { discr, targets }
                if let Some(place) = discr.place()
                    && self.is_ssa(place)
                    // Reduce the potential compile-time overhead.
                    && targets.all_targets().len() < 16 =>
            {
                let mut distinct_targets: FxHashMap<BasicBlock, u64> = FxHashMap::default();
                for (_, target) in targets.iter() {
                    let targets = distinct_targets.entry(target).or_default();
                    *targets += 1;
                }
                for (val, target) in targets.iter() {
                    if distinct_targets[&target] != 1 {
                        // FIXME: For multiple targets, the range can be the union of their values.
                        continue;
                    }
                    let successor = Location { block: target, statement_index: 0 };
                    if self.unique_predecessors.contains(successor.block) {
                        assert_ne!(location.block, successor.block);
                        let range = WrappingRange { start: val, end: val };
                        self.insert_range(place, successor, range);
                        self.insert_comparison_range(place, successor, val != 0);
                    }
                }

                // FIXME: The range for the otherwise target be extend to more types.
                // For instance, `val` is within the range [4, 1) at the otherwise target of `matches!(val, 1 | 2 | 3)`.
                let otherwise = Location { block: targets.otherwise(), statement_index: 0 };
                if place.ty(self.local_decls, self.tcx).ty.is_bool()
                    && let [val] = targets.all_values()
                    && self.unique_predecessors.contains(otherwise.block)
                {
                    assert_ne!(location.block, otherwise.block);
                    let range = if val.get() == 0 {
                        WrappingRange { start: 1, end: 1 }
                    } else {
                        WrappingRange { start: 0, end: 0 }
                    };
                    let expected = range.start != 0;
                    self.insert_range(place, otherwise, range);
                    self.insert_comparison_range(place, otherwise, expected);
                }
            }
            _ => {}
        }
    }
}
