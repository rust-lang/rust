/// Functionality for terminators and helper types that appear in terminators.
use rustc_hir::LangItem;
use smallvec::SmallVec;

use super::{BasicBlock, InlineAsmOperand, Operand, SourceInfo, TerminatorKind, UnwindAction};
use rustc_macros::HashStable;
use std::iter;
use std::slice;

use super::*;

impl SwitchTargets {
    /// Creates switch targets from an iterator of values and target blocks.
    ///
    /// The iterator may be empty, in which case the `SwitchInt` instruction is equivalent to
    /// `goto otherwise;`.
    pub fn new(targets: impl Iterator<Item = (u128, BasicBlock)>, otherwise: BasicBlock) -> Self {
        let (values, mut targets): (SmallVec<_>, SmallVec<_>) = targets.unzip();
        targets.push(otherwise);
        Self { values, targets }
    }

    /// Builds a switch targets definition that jumps to `then` if the tested value equals `value`,
    /// and to `else_` if not.
    pub fn static_if(value: u128, then: BasicBlock, else_: BasicBlock) -> Self {
        Self { values: smallvec![value], targets: smallvec![then, else_] }
    }

    /// Inverse of `SwitchTargets::static_if`.
    pub fn as_static_if(&self) -> Option<(u128, BasicBlock, BasicBlock)> {
        if let &[value] = &self.values[..]
            && let &[then, else_] = &self.targets[..]
        {
            Some((value, then, else_))
        } else {
            None
        }
    }

    /// Returns the fallback target that is jumped to when none of the values match the operand.
    pub fn otherwise(&self) -> BasicBlock {
        *self.targets.last().unwrap()
    }

    /// Returns an iterator over the switch targets.
    ///
    /// The iterator will yield tuples containing the value and corresponding target to jump to, not
    /// including the `otherwise` fallback target.
    ///
    /// Note that this may yield 0 elements. Only the `otherwise` branch is mandatory.
    pub fn iter(&self) -> SwitchTargetsIter<'_> {
        SwitchTargetsIter { inner: iter::zip(&self.values, &self.targets) }
    }

    /// Returns a slice with all possible jump targets (including the fallback target).
    pub fn all_targets(&self) -> &[BasicBlock] {
        &self.targets
    }

    pub fn all_targets_mut(&mut self) -> &mut [BasicBlock] {
        &mut self.targets
    }

    /// Finds the `BasicBlock` to which this `SwitchInt` will branch given the
    /// specific value. This cannot fail, as it'll return the `otherwise`
    /// branch if there's not a specific match for the value.
    pub fn target_for_value(&self, value: u128) -> BasicBlock {
        self.iter().find_map(|(v, t)| (v == value).then_some(t)).unwrap_or_else(|| self.otherwise())
    }
}

pub struct SwitchTargetsIter<'a> {
    inner: iter::Zip<slice::Iter<'a, u128>, slice::Iter<'a, BasicBlock>>,
}

impl<'a> Iterator for SwitchTargetsIter<'a> {
    type Item = (u128, BasicBlock);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(val, bb)| (*val, *bb))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a> ExactSizeIterator for SwitchTargetsIter<'a> {}

impl UnwindAction {
    fn cleanup_block(self) -> Option<BasicBlock> {
        match self {
            UnwindAction::Cleanup(bb) => Some(bb),
            UnwindAction::Continue | UnwindAction::Unreachable | UnwindAction::Terminate(_) => None,
        }
    }
}

impl UnwindTerminateReason {
    pub fn as_str(self) -> &'static str {
        // Keep this in sync with the messages in `core/src/panicking.rs`.
        match self {
            UnwindTerminateReason::Abi => "panic in a function that cannot unwind",
            UnwindTerminateReason::InCleanup => "panic in a destructor during cleanup",
        }
    }

    /// A short representation of this used for MIR printing.
    pub fn as_short_str(self) -> &'static str {
        match self {
            UnwindTerminateReason::Abi => "abi",
            UnwindTerminateReason::InCleanup => "cleanup",
        }
    }

    pub fn lang_item(self) -> LangItem {
        match self {
            UnwindTerminateReason::Abi => LangItem::PanicCannotUnwind,
            UnwindTerminateReason::InCleanup => LangItem::PanicInCleanup,
        }
    }
}

impl<O> AssertKind<O> {
    /// Returns true if this an overflow checking assertion controlled by -C overflow-checks.
    pub fn is_optional_overflow_check(&self) -> bool {
        use AssertKind::*;
        use BinOp::*;
        matches!(self, OverflowNeg(..) | Overflow(Add | Sub | Mul | Shl | Shr, ..))
    }

    /// Get the message that is printed at runtime when this assertion fails.
    ///
    /// The caller is expected to handle `BoundsCheck` and `MisalignedPointerDereference` by
    /// invoking the appropriate lang item (panic_bounds_check/panic_misaligned_pointer_dereference)
    /// instead of printing a static message.
    pub fn description(&self) -> &'static str {
        use AssertKind::*;
        match self {
            Overflow(BinOp::Add, _, _) => "attempt to add with overflow",
            Overflow(BinOp::Sub, _, _) => "attempt to subtract with overflow",
            Overflow(BinOp::Mul, _, _) => "attempt to multiply with overflow",
            Overflow(BinOp::Div, _, _) => "attempt to divide with overflow",
            Overflow(BinOp::Rem, _, _) => "attempt to calculate the remainder with overflow",
            OverflowNeg(_) => "attempt to negate with overflow",
            Overflow(BinOp::Shr, _, _) => "attempt to shift right with overflow",
            Overflow(BinOp::Shl, _, _) => "attempt to shift left with overflow",
            Overflow(op, _, _) => bug!("{:?} cannot overflow", op),
            DivisionByZero(_) => "attempt to divide by zero",
            RemainderByZero(_) => "attempt to calculate the remainder with a divisor of zero",
            ResumedAfterReturn(CoroutineKind::Coroutine) => "coroutine resumed after completion",
            ResumedAfterReturn(CoroutineKind::Async(_)) => "`async fn` resumed after completion",
            ResumedAfterReturn(CoroutineKind::Gen(_)) => {
                "`gen fn` should just keep returning `None` after completion"
            }
            ResumedAfterPanic(CoroutineKind::Coroutine) => "coroutine resumed after panicking",
            ResumedAfterPanic(CoroutineKind::Async(_)) => "`async fn` resumed after panicking",
            ResumedAfterPanic(CoroutineKind::Gen(_)) => {
                "`gen fn` should just keep returning `None` after panicking"
            }

            BoundsCheck { .. } | MisalignedPointerDereference { .. } => {
                bug!("Unexpected AssertKind")
            }
        }
    }

    /// Format the message arguments for the `assert(cond, msg..)` terminator in MIR printing.
    ///
    /// Needs to be kept in sync with the run-time behavior (which is defined by
    /// `AssertKind::description` and the lang items mentioned in its docs).
    /// Note that we deliberately show more details here than we do at runtime, such as the actual
    /// numbers that overflowed -- it is much easier to do so here than at runtime.
    pub fn fmt_assert_args<W: fmt::Write>(&self, f: &mut W) -> fmt::Result
    where
        O: Debug,
    {
        use AssertKind::*;
        match self {
            BoundsCheck { ref len, ref index } => write!(
                f,
                "\"index out of bounds: the length is {{}} but the index is {{}}\", {len:?}, {index:?}"
            ),

            OverflowNeg(op) => {
                write!(f, "\"attempt to negate `{{}}`, which would overflow\", {op:?}")
            }
            DivisionByZero(op) => write!(f, "\"attempt to divide `{{}}` by zero\", {op:?}"),
            RemainderByZero(op) => write!(
                f,
                "\"attempt to calculate the remainder of `{{}}` with a divisor of zero\", {op:?}"
            ),
            Overflow(BinOp::Add, l, r) => write!(
                f,
                "\"attempt to compute `{{}} + {{}}`, which would overflow\", {l:?}, {r:?}"
            ),
            Overflow(BinOp::Sub, l, r) => write!(
                f,
                "\"attempt to compute `{{}} - {{}}`, which would overflow\", {l:?}, {r:?}"
            ),
            Overflow(BinOp::Mul, l, r) => write!(
                f,
                "\"attempt to compute `{{}} * {{}}`, which would overflow\", {l:?}, {r:?}"
            ),
            Overflow(BinOp::Div, l, r) => write!(
                f,
                "\"attempt to compute `{{}} / {{}}`, which would overflow\", {l:?}, {r:?}"
            ),
            Overflow(BinOp::Rem, l, r) => write!(
                f,
                "\"attempt to compute the remainder of `{{}} % {{}}`, which would overflow\", {l:?}, {r:?}"
            ),
            Overflow(BinOp::Shr, _, r) => {
                write!(f, "\"attempt to shift right by `{{}}`, which would overflow\", {r:?}")
            }
            Overflow(BinOp::Shl, _, r) => {
                write!(f, "\"attempt to shift left by `{{}}`, which would overflow\", {r:?}")
            }
            MisalignedPointerDereference { required, found } => {
                write!(
                    f,
                    "\"misaligned pointer dereference: address must be a multiple of {{}} but is {{}}\", {required:?}, {found:?}"
                )
            }
            _ => write!(f, "\"{}\"", self.description()),
        }
    }

    /// Format the diagnostic message for use in a lint (e.g. when the assertion fails during const-eval).
    ///
    /// Needs to be kept in sync with the run-time behavior (which is defined by
    /// `AssertKind::description` and the lang items mentioned in its docs).
    /// Note that we deliberately show more details here than we do at runtime, such as the actual
    /// numbers that overflowed -- it is much easier to do so here than at runtime.
    pub fn diagnostic_message(&self) -> DiagnosticMessage {
        use crate::fluent_generated::*;
        use AssertKind::*;

        match self {
            BoundsCheck { .. } => middle_bounds_check,
            Overflow(BinOp::Shl, _, _) => middle_assert_shl_overflow,
            Overflow(BinOp::Shr, _, _) => middle_assert_shr_overflow,
            Overflow(_, _, _) => middle_assert_op_overflow,
            OverflowNeg(_) => middle_assert_overflow_neg,
            DivisionByZero(_) => middle_assert_divide_by_zero,
            RemainderByZero(_) => middle_assert_remainder_by_zero,
            ResumedAfterReturn(CoroutineKind::Async(_)) => middle_assert_async_resume_after_return,
            ResumedAfterReturn(CoroutineKind::Gen(_)) => {
                bug!("gen blocks can be resumed after they return and will keep returning `None`")
            }
            ResumedAfterReturn(CoroutineKind::Coroutine) => {
                middle_assert_coroutine_resume_after_return
            }
            ResumedAfterPanic(CoroutineKind::Async(_)) => middle_assert_async_resume_after_panic,
            ResumedAfterPanic(CoroutineKind::Gen(_)) => middle_assert_gen_resume_after_panic,
            ResumedAfterPanic(CoroutineKind::Coroutine) => {
                middle_assert_coroutine_resume_after_panic
            }

            MisalignedPointerDereference { .. } => middle_assert_misaligned_ptr_deref,
        }
    }

    pub fn add_args(self, adder: &mut dyn FnMut(Cow<'static, str>, DiagnosticArgValue<'static>))
    where
        O: fmt::Debug,
    {
        use AssertKind::*;

        macro_rules! add {
            ($name: expr, $value: expr) => {
                adder($name.into(), $value.into_diagnostic_arg());
            };
        }

        match self {
            BoundsCheck { len, index } => {
                add!("len", format!("{len:?}"));
                add!("index", format!("{index:?}"));
            }
            Overflow(BinOp::Shl | BinOp::Shr, _, val)
            | DivisionByZero(val)
            | RemainderByZero(val)
            | OverflowNeg(val) => {
                add!("val", format!("{val:#?}"));
            }
            Overflow(binop, left, right) => {
                add!("op", binop.to_hir_binop().as_str());
                add!("left", format!("{left:#?}"));
                add!("right", format!("{right:#?}"));
            }
            ResumedAfterReturn(_) | ResumedAfterPanic(_) => {}
            MisalignedPointerDereference { required, found } => {
                add!("required", format!("{required:#?}"));
                add!("found", format!("{found:#?}"));
            }
        }
    }
}

#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct Terminator<'tcx> {
    pub source_info: SourceInfo,
    pub kind: TerminatorKind<'tcx>,
}

pub type Successors<'a> = impl DoubleEndedIterator<Item = BasicBlock> + 'a;
pub type SuccessorsMut<'a> =
    iter::Chain<std::option::IntoIter<&'a mut BasicBlock>, slice::IterMut<'a, BasicBlock>>;

impl<'tcx> Terminator<'tcx> {
    pub fn successors(&self) -> Successors<'_> {
        self.kind.successors()
    }

    pub fn successors_mut(&mut self) -> SuccessorsMut<'_> {
        self.kind.successors_mut()
    }

    pub fn unwind(&self) -> Option<&UnwindAction> {
        self.kind.unwind()
    }

    pub fn unwind_mut(&mut self) -> Option<&mut UnwindAction> {
        self.kind.unwind_mut()
    }
}

impl<'tcx> TerminatorKind<'tcx> {
    pub fn if_(cond: Operand<'tcx>, t: BasicBlock, f: BasicBlock) -> TerminatorKind<'tcx> {
        TerminatorKind::SwitchInt { discr: cond, targets: SwitchTargets::static_if(0, f, t) }
    }

    pub fn successors(&self) -> Successors<'_> {
        use self::TerminatorKind::*;
        match *self {
            Call { target: Some(t), unwind: UnwindAction::Cleanup(ref u), .. }
            | Yield { resume: t, drop: Some(ref u), .. }
            | Drop { target: t, unwind: UnwindAction::Cleanup(ref u), .. }
            | Assert { target: t, unwind: UnwindAction::Cleanup(ref u), .. }
            | FalseUnwind { real_target: t, unwind: UnwindAction::Cleanup(ref u) }
            | InlineAsm { destination: Some(t), unwind: UnwindAction::Cleanup(ref u), .. } => {
                Some(t).into_iter().chain(slice::from_ref(u).into_iter().copied())
            }
            Goto { target: t }
            | Call { target: None, unwind: UnwindAction::Cleanup(t), .. }
            | Call { target: Some(t), unwind: _, .. }
            | Yield { resume: t, drop: None, .. }
            | Drop { target: t, unwind: _, .. }
            | Assert { target: t, unwind: _, .. }
            | FalseUnwind { real_target: t, unwind: _ }
            | InlineAsm { destination: None, unwind: UnwindAction::Cleanup(t), .. }
            | InlineAsm { destination: Some(t), unwind: _, .. } => {
                Some(t).into_iter().chain((&[]).into_iter().copied())
            }
            UnwindResume
            | UnwindTerminate(_)
            | CoroutineDrop
            | Return
            | Unreachable
            | Call { target: None, unwind: _, .. }
            | InlineAsm { destination: None, unwind: _, .. } => {
                None.into_iter().chain((&[]).into_iter().copied())
            }
            SwitchInt { ref targets, .. } => {
                None.into_iter().chain(targets.targets.iter().copied())
            }
            FalseEdge { real_target, ref imaginary_target } => Some(real_target)
                .into_iter()
                .chain(slice::from_ref(imaginary_target).into_iter().copied()),
        }
    }

    pub fn successors_mut(&mut self) -> SuccessorsMut<'_> {
        use self::TerminatorKind::*;
        match *self {
            Call { target: Some(ref mut t), unwind: UnwindAction::Cleanup(ref mut u), .. }
            | Yield { resume: ref mut t, drop: Some(ref mut u), .. }
            | Drop { target: ref mut t, unwind: UnwindAction::Cleanup(ref mut u), .. }
            | Assert { target: ref mut t, unwind: UnwindAction::Cleanup(ref mut u), .. }
            | FalseUnwind { real_target: ref mut t, unwind: UnwindAction::Cleanup(ref mut u) }
            | InlineAsm {
                destination: Some(ref mut t),
                unwind: UnwindAction::Cleanup(ref mut u),
                ..
            } => Some(t).into_iter().chain(slice::from_mut(u)),
            Goto { target: ref mut t }
            | Call { target: None, unwind: UnwindAction::Cleanup(ref mut t), .. }
            | Call { target: Some(ref mut t), unwind: _, .. }
            | Yield { resume: ref mut t, drop: None, .. }
            | Drop { target: ref mut t, unwind: _, .. }
            | Assert { target: ref mut t, unwind: _, .. }
            | FalseUnwind { real_target: ref mut t, unwind: _ }
            | InlineAsm { destination: None, unwind: UnwindAction::Cleanup(ref mut t), .. }
            | InlineAsm { destination: Some(ref mut t), unwind: _, .. } => {
                Some(t).into_iter().chain(&mut [])
            }
            UnwindResume
            | UnwindTerminate(_)
            | CoroutineDrop
            | Return
            | Unreachable
            | Call { target: None, unwind: _, .. }
            | InlineAsm { destination: None, unwind: _, .. } => None.into_iter().chain(&mut []),
            SwitchInt { ref mut targets, .. } => None.into_iter().chain(&mut targets.targets),
            FalseEdge { ref mut real_target, ref mut imaginary_target } => {
                Some(real_target).into_iter().chain(slice::from_mut(imaginary_target))
            }
        }
    }

    pub fn unwind(&self) -> Option<&UnwindAction> {
        match *self {
            TerminatorKind::Goto { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::Yield { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::FalseEdge { .. } => None,
            TerminatorKind::Call { ref unwind, .. }
            | TerminatorKind::Assert { ref unwind, .. }
            | TerminatorKind::Drop { ref unwind, .. }
            | TerminatorKind::FalseUnwind { ref unwind, .. }
            | TerminatorKind::InlineAsm { ref unwind, .. } => Some(unwind),
        }
    }

    pub fn unwind_mut(&mut self) -> Option<&mut UnwindAction> {
        match *self {
            TerminatorKind::Goto { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::Yield { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::FalseEdge { .. } => None,
            TerminatorKind::Call { ref mut unwind, .. }
            | TerminatorKind::Assert { ref mut unwind, .. }
            | TerminatorKind::Drop { ref mut unwind, .. }
            | TerminatorKind::FalseUnwind { ref mut unwind, .. }
            | TerminatorKind::InlineAsm { ref mut unwind, .. } => Some(unwind),
        }
    }

    pub fn as_switch(&self) -> Option<(&Operand<'tcx>, &SwitchTargets)> {
        match self {
            TerminatorKind::SwitchInt { discr, targets } => Some((discr, targets)),
            _ => None,
        }
    }

    pub fn as_goto(&self) -> Option<BasicBlock> {
        match self {
            TerminatorKind::Goto { target } => Some(*target),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum TerminatorEdges<'mir, 'tcx> {
    /// For terminators that have no successor, like `return`.
    None,
    /// For terminators that a single successor, like `goto`, and `assert` without cleanup block.
    Single(BasicBlock),
    /// For terminators that two successors, `assert` with cleanup block and `falseEdge`.
    Double(BasicBlock, BasicBlock),
    /// Special action for `Yield`, `Call` and `InlineAsm` terminators.
    AssignOnReturn {
        return_: Option<BasicBlock>,
        /// The cleanup block, if it exists.
        cleanup: Option<BasicBlock>,
        place: CallReturnPlaces<'mir, 'tcx>,
    },
    /// Special edge for `SwitchInt`.
    SwitchInt { targets: &'mir SwitchTargets, discr: &'mir Operand<'tcx> },
}

/// List of places that are written to after a successful (non-unwind) return
/// from a `Call`, `Yield` or `InlineAsm`.
#[derive(Copy, Clone, Debug)]
pub enum CallReturnPlaces<'a, 'tcx> {
    Call(Place<'tcx>),
    Yield(Place<'tcx>),
    InlineAsm(&'a [InlineAsmOperand<'tcx>]),
}

impl<'tcx> CallReturnPlaces<'_, 'tcx> {
    pub fn for_each(&self, mut f: impl FnMut(Place<'tcx>)) {
        match *self {
            Self::Call(place) | Self::Yield(place) => f(place),
            Self::InlineAsm(operands) => {
                for op in operands {
                    match *op {
                        InlineAsmOperand::Out { place: Some(place), .. }
                        | InlineAsmOperand::InOut { out_place: Some(place), .. } => f(place),
                        _ => {}
                    }
                }
            }
        }
    }
}

impl<'tcx> Terminator<'tcx> {
    pub fn edges(&self) -> TerminatorEdges<'_, 'tcx> {
        self.kind.edges()
    }
}

impl<'tcx> TerminatorKind<'tcx> {
    pub fn edges(&self) -> TerminatorEdges<'_, 'tcx> {
        use TerminatorKind::*;
        match *self {
            Return | UnwindResume | UnwindTerminate(_) | CoroutineDrop | Unreachable => {
                TerminatorEdges::None
            }

            Goto { target } => TerminatorEdges::Single(target),

            Assert { target, unwind, expected: _, msg: _, cond: _ }
            | Drop { target, unwind, place: _, replace: _ }
            | FalseUnwind { real_target: target, unwind } => match unwind {
                UnwindAction::Cleanup(unwind) => TerminatorEdges::Double(target, unwind),
                UnwindAction::Continue | UnwindAction::Terminate(_) | UnwindAction::Unreachable => {
                    TerminatorEdges::Single(target)
                }
            },

            FalseEdge { real_target, imaginary_target } => {
                TerminatorEdges::Double(real_target, imaginary_target)
            }

            Yield { resume: target, drop, resume_arg, value: _ } => {
                TerminatorEdges::AssignOnReturn {
                    return_: Some(target),
                    cleanup: drop,
                    place: CallReturnPlaces::Yield(resume_arg),
                }
            }

            Call { unwind, destination, target, func: _, args: _, fn_span: _, call_source: _ } => {
                TerminatorEdges::AssignOnReturn {
                    return_: target,
                    cleanup: unwind.cleanup_block(),
                    place: CallReturnPlaces::Call(destination),
                }
            }

            InlineAsm {
                template: _,
                ref operands,
                options: _,
                line_spans: _,
                destination,
                unwind,
            } => TerminatorEdges::AssignOnReturn {
                return_: destination,
                cleanup: unwind.cleanup_block(),
                place: CallReturnPlaces::InlineAsm(operands),
            },

            SwitchInt { ref targets, ref discr } => TerminatorEdges::SwitchInt { targets, discr },
        }
    }
}
