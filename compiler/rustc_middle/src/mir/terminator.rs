//! Functionality for terminators and helper types that appear in terminators.

use std::slice;

use rustc_ast::InlineAsmOptions;
use rustc_data_structures::packed::Pu128;
use rustc_hir::LangItem;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use smallvec::{SmallVec, smallvec};

use super::*;

impl SwitchTargets {
    /// Creates switch targets from an iterator of values and target blocks.
    ///
    /// The iterator may be empty, in which case the `SwitchInt` instruction is equivalent to
    /// `goto otherwise;`.
    pub fn new(targets: impl Iterator<Item = (u128, BasicBlock)>, otherwise: BasicBlock) -> Self {
        let (values, mut targets): (SmallVec<_>, SmallVec<_>) =
            targets.map(|(v, t)| (Pu128(v), t)).unzip();
        targets.push(otherwise);
        Self { values, targets }
    }

    /// Builds a switch targets definition that jumps to `then` if the tested value equals `value`,
    /// and to `else_` if not.
    pub fn static_if(value: u128, then: BasicBlock, else_: BasicBlock) -> Self {
        Self { values: smallvec![Pu128(value)], targets: smallvec![then, else_] }
    }

    /// Inverse of `SwitchTargets::static_if`.
    #[inline]
    pub fn as_static_if(&self) -> Option<(u128, BasicBlock, BasicBlock)> {
        if let &[value] = &self.values[..]
            && let &[then, else_] = &self.targets[..]
        {
            Some((value.get(), then, else_))
        } else {
            None
        }
    }

    /// Returns the fallback target that is jumped to when none of the values match the operand.
    #[inline]
    pub fn otherwise(&self) -> BasicBlock {
        *self.targets.last().unwrap()
    }

    /// Returns an iterator over the switch targets.
    ///
    /// The iterator will yield tuples containing the value and corresponding target to jump to, not
    /// including the `otherwise` fallback target.
    ///
    /// Note that this may yield 0 elements. Only the `otherwise` branch is mandatory.
    #[inline]
    pub fn iter(&self) -> SwitchTargetsIter<'_> {
        SwitchTargetsIter { inner: iter::zip(&self.values, &self.targets) }
    }

    /// Returns a slice with all possible jump targets (including the fallback target).
    #[inline]
    pub fn all_targets(&self) -> &[BasicBlock] {
        &self.targets
    }

    #[inline]
    pub fn all_targets_mut(&mut self) -> &mut [BasicBlock] {
        &mut self.targets
    }

    /// Returns a slice with all considered values (not including the fallback).
    #[inline]
    pub fn all_values(&self) -> &[Pu128] {
        &self.values
    }

    #[inline]
    pub fn all_values_mut(&mut self) -> &mut [Pu128] {
        &mut self.values
    }

    /// Finds the `BasicBlock` to which this `SwitchInt` will branch given the
    /// specific value. This cannot fail, as it'll return the `otherwise`
    /// branch if there's not a specific match for the value.
    #[inline]
    pub fn target_for_value(&self, value: u128) -> BasicBlock {
        self.iter().find_map(|(v, t)| (v == value).then_some(t)).unwrap_or_else(|| self.otherwise())
    }

    /// Adds a new target to the switch. Panics if you add an already present value.
    #[inline]
    pub fn add_target(&mut self, value: u128, bb: BasicBlock) {
        let value = Pu128(value);
        if self.values.contains(&value) {
            bug!("target value {:?} already present", value);
        }
        self.values.push(value);
        self.targets.insert(self.targets.len() - 1, bb);
    }

    /// Returns true if all targets (including the fallback target) are distinct.
    #[inline]
    pub fn is_distinct(&self) -> bool {
        self.targets.iter().collect::<FxHashSet<_>>().len() == self.targets.len()
    }
}

pub struct SwitchTargetsIter<'a> {
    inner: iter::Zip<slice::Iter<'a, Pu128>, slice::Iter<'a, BasicBlock>>,
}

impl<'a> Iterator for SwitchTargetsIter<'a> {
    type Item = (u128, BasicBlock);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(val, bb)| (val.get(), *bb))
    }

    #[inline]
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

    /// Get the lang item that is invoked to print a static message when this assert fires.
    ///
    /// The caller is expected to handle `BoundsCheck` and `MisalignedPointerDereference` by
    /// invoking the appropriate lang item (panic_bounds_check/panic_misaligned_pointer_dereference)
    /// instead of printing a static message. Those have dynamic arguments that aren't present for
    /// the rest of the messages here.
    pub fn panic_function(&self) -> LangItem {
        use AssertKind::*;
        match self {
            Overflow(BinOp::Add, _, _) => LangItem::PanicAddOverflow,
            Overflow(BinOp::Sub, _, _) => LangItem::PanicSubOverflow,
            Overflow(BinOp::Mul, _, _) => LangItem::PanicMulOverflow,
            Overflow(BinOp::Div, _, _) => LangItem::PanicDivOverflow,
            Overflow(BinOp::Rem, _, _) => LangItem::PanicRemOverflow,
            OverflowNeg(_) => LangItem::PanicNegOverflow,
            Overflow(BinOp::Shr, _, _) => LangItem::PanicShrOverflow,
            Overflow(BinOp::Shl, _, _) => LangItem::PanicShlOverflow,
            Overflow(op, _, _) => bug!("{:?} cannot overflow", op),
            DivisionByZero(_) => LangItem::PanicDivZero,
            RemainderByZero(_) => LangItem::PanicRemZero,
            ResumedAfterReturn(CoroutineKind::Coroutine(_)) => LangItem::PanicCoroutineResumed,
            ResumedAfterReturn(CoroutineKind::Desugared(CoroutineDesugaring::Async, _)) => {
                LangItem::PanicAsyncFnResumed
            }
            ResumedAfterReturn(CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen, _)) => {
                LangItem::PanicAsyncGenFnResumed
            }
            ResumedAfterReturn(CoroutineKind::Desugared(CoroutineDesugaring::Gen, _)) => {
                LangItem::PanicGenFnNone
            }
            ResumedAfterPanic(CoroutineKind::Coroutine(_)) => LangItem::PanicCoroutineResumedPanic,
            ResumedAfterPanic(CoroutineKind::Desugared(CoroutineDesugaring::Async, _)) => {
                LangItem::PanicAsyncFnResumedPanic
            }
            ResumedAfterPanic(CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen, _)) => {
                LangItem::PanicAsyncGenFnResumedPanic
            }
            ResumedAfterPanic(CoroutineKind::Desugared(CoroutineDesugaring::Gen, _)) => {
                LangItem::PanicGenFnNonePanic
            }
            NullPointerDereference => LangItem::PanicNullPointerDereference,

            BoundsCheck { .. } | MisalignedPointerDereference { .. } => {
                bug!("Unexpected AssertKind")
            }
        }
    }

    /// Format the message arguments for the `assert(cond, msg..)` terminator in MIR printing.
    ///
    /// Needs to be kept in sync with the run-time behavior (which is defined by
    /// `AssertKind::panic_function` and the lang items mentioned in its docs).
    /// Note that we deliberately show more details here than we do at runtime, such as the actual
    /// numbers that overflowed -- it is much easier to do so here than at runtime.
    pub fn fmt_assert_args<W: fmt::Write>(&self, f: &mut W) -> fmt::Result
    where
        O: Debug,
    {
        use AssertKind::*;
        match self {
            BoundsCheck { len, index } => write!(
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
            Overflow(op, _, _) => bug!("{:?} cannot overflow", op),
            MisalignedPointerDereference { required, found } => {
                write!(
                    f,
                    "\"misaligned pointer dereference: address must be a multiple of {{}} but is {{}}\", {required:?}, {found:?}"
                )
            }
            NullPointerDereference => write!(f, "\"null pointer dereference occurred\""),
            ResumedAfterReturn(CoroutineKind::Coroutine(_)) => {
                write!(f, "\"coroutine resumed after completion\"")
            }
            ResumedAfterReturn(CoroutineKind::Desugared(CoroutineDesugaring::Async, _)) => {
                write!(f, "\"`async fn` resumed after completion\"")
            }
            ResumedAfterReturn(CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen, _)) => {
                write!(f, "\"`async gen fn` resumed after completion\"")
            }
            ResumedAfterReturn(CoroutineKind::Desugared(CoroutineDesugaring::Gen, _)) => {
                write!(f, "\"`gen fn` should just keep returning `None` after completion\"")
            }
            ResumedAfterPanic(CoroutineKind::Coroutine(_)) => {
                write!(f, "\"coroutine resumed after panicking\"")
            }
            ResumedAfterPanic(CoroutineKind::Desugared(CoroutineDesugaring::Async, _)) => {
                write!(f, "\"`async fn` resumed after panicking\"")
            }
            ResumedAfterPanic(CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen, _)) => {
                write!(f, "\"`async gen fn` resumed after panicking\"")
            }
            ResumedAfterPanic(CoroutineKind::Desugared(CoroutineDesugaring::Gen, _)) => {
                write!(f, "\"`gen fn` should just keep returning `None` after panicking\"")
            }
        }
    }

    /// Format the diagnostic message for use in a lint (e.g. when the assertion fails during const-eval).
    ///
    /// Needs to be kept in sync with the run-time behavior (which is defined by
    /// `AssertKind::panic_function` and the lang items mentioned in its docs).
    /// Note that we deliberately show more details here than we do at runtime, such as the actual
    /// numbers that overflowed -- it is much easier to do so here than at runtime.
    pub fn diagnostic_message(&self) -> DiagMessage {
        use AssertKind::*;

        use crate::fluent_generated::*;

        match self {
            BoundsCheck { .. } => middle_bounds_check,
            Overflow(BinOp::Shl, _, _) => middle_assert_shl_overflow,
            Overflow(BinOp::Shr, _, _) => middle_assert_shr_overflow,
            Overflow(_, _, _) => middle_assert_op_overflow,
            OverflowNeg(_) => middle_assert_overflow_neg,
            DivisionByZero(_) => middle_assert_divide_by_zero,
            RemainderByZero(_) => middle_assert_remainder_by_zero,
            ResumedAfterReturn(CoroutineKind::Desugared(CoroutineDesugaring::Async, _)) => {
                middle_assert_async_resume_after_return
            }
            ResumedAfterReturn(CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen, _)) => {
                todo!()
            }
            ResumedAfterReturn(CoroutineKind::Desugared(CoroutineDesugaring::Gen, _)) => {
                bug!("gen blocks can be resumed after they return and will keep returning `None`")
            }
            ResumedAfterReturn(CoroutineKind::Coroutine(_)) => {
                middle_assert_coroutine_resume_after_return
            }
            ResumedAfterPanic(CoroutineKind::Desugared(CoroutineDesugaring::Async, _)) => {
                middle_assert_async_resume_after_panic
            }
            ResumedAfterPanic(CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen, _)) => {
                todo!()
            }
            ResumedAfterPanic(CoroutineKind::Desugared(CoroutineDesugaring::Gen, _)) => {
                middle_assert_gen_resume_after_panic
            }
            ResumedAfterPanic(CoroutineKind::Coroutine(_)) => {
                middle_assert_coroutine_resume_after_panic
            }
            NullPointerDereference => middle_assert_null_ptr_deref,
            MisalignedPointerDereference { .. } => middle_assert_misaligned_ptr_deref,
        }
    }

    pub fn add_args(self, adder: &mut dyn FnMut(DiagArgName, DiagArgValue))
    where
        O: fmt::Debug,
    {
        use AssertKind::*;

        macro_rules! add {
            ($name: expr, $value: expr) => {
                adder($name.into(), $value.into_diag_arg(&mut None));
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
            ResumedAfterReturn(_) | ResumedAfterPanic(_) | NullPointerDereference => {}
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

impl<'tcx> Terminator<'tcx> {
    #[inline]
    pub fn successors(&self) -> Successors<'_> {
        self.kind.successors()
    }

    #[inline]
    pub fn successors_mut(&mut self) -> SuccessorsMut<'_> {
        self.kind.successors_mut()
    }

    #[inline]
    pub fn unwind(&self) -> Option<&UnwindAction> {
        self.kind.unwind()
    }

    #[inline]
    pub fn unwind_mut(&mut self) -> Option<&mut UnwindAction> {
        self.kind.unwind_mut()
    }
}

impl<'tcx> TerminatorKind<'tcx> {
    /// Returns a simple string representation of a `TerminatorKind` variant, independent of any
    /// values it might hold (e.g. `TerminatorKind::Call` always returns `"Call"`).
    pub const fn name(&self) -> &'static str {
        match self {
            TerminatorKind::Goto { .. } => "Goto",
            TerminatorKind::SwitchInt { .. } => "SwitchInt",
            TerminatorKind::UnwindResume => "UnwindResume",
            TerminatorKind::UnwindTerminate(_) => "UnwindTerminate",
            TerminatorKind::Return => "Return",
            TerminatorKind::Unreachable => "Unreachable",
            TerminatorKind::Drop { .. } => "Drop",
            TerminatorKind::Call { .. } => "Call",
            TerminatorKind::TailCall { .. } => "TailCall",
            TerminatorKind::Assert { .. } => "Assert",
            TerminatorKind::Yield { .. } => "Yield",
            TerminatorKind::CoroutineDrop => "CoroutineDrop",
            TerminatorKind::FalseEdge { .. } => "FalseEdge",
            TerminatorKind::FalseUnwind { .. } => "FalseUnwind",
            TerminatorKind::InlineAsm { .. } => "InlineAsm",
        }
    }

    #[inline]
    pub fn if_(cond: Operand<'tcx>, t: BasicBlock, f: BasicBlock) -> TerminatorKind<'tcx> {
        TerminatorKind::SwitchInt { discr: cond, targets: SwitchTargets::static_if(0, f, t) }
    }
}

pub use helper::*;

mod helper {
    use super::*;
    pub type Successors<'a> = impl DoubleEndedIterator<Item = BasicBlock> + 'a;
    pub type SuccessorsMut<'a> = impl DoubleEndedIterator<Item = &'a mut BasicBlock> + 'a;

    impl SwitchTargets {
        /// Like [`SwitchTargets::target_for_value`], but returning the same type as
        /// [`Terminator::successors`].
        #[inline]
        #[define_opaque(Successors)]
        pub fn successors_for_value(&self, value: u128) -> Successors<'_> {
            let target = self.target_for_value(value);
            (&[]).into_iter().copied().chain(Some(target))
        }
    }

    impl<'tcx> TerminatorKind<'tcx> {
        #[inline]
        #[define_opaque(Successors)]
        pub fn successors(&self) -> Successors<'_> {
            use self::TerminatorKind::*;
            match *self {
                Call { target: Some(ref t), unwind: UnwindAction::Cleanup(u), .. }
                | Yield { resume: ref t, drop: Some(u), .. }
                | Drop { target: ref t, unwind: UnwindAction::Cleanup(u), .. }
                | Assert { target: ref t, unwind: UnwindAction::Cleanup(u), .. }
                | FalseUnwind { real_target: ref t, unwind: UnwindAction::Cleanup(u) } => {
                    slice::from_ref(t).into_iter().copied().chain(Some(u))
                }
                Goto { target: ref t }
                | Call { target: None, unwind: UnwindAction::Cleanup(ref t), .. }
                | Call { target: Some(ref t), unwind: _, .. }
                | Yield { resume: ref t, drop: None, .. }
                | Drop { target: ref t, unwind: _, .. }
                | Assert { target: ref t, unwind: _, .. }
                | FalseUnwind { real_target: ref t, unwind: _ } => {
                    slice::from_ref(t).into_iter().copied().chain(None)
                }
                UnwindResume
                | UnwindTerminate(_)
                | CoroutineDrop
                | Return
                | Unreachable
                | TailCall { .. }
                | Call { target: None, unwind: _, .. } => (&[]).into_iter().copied().chain(None),
                InlineAsm { ref targets, unwind: UnwindAction::Cleanup(u), .. } => {
                    targets.iter().copied().chain(Some(u))
                }
                InlineAsm { ref targets, unwind: _, .. } => targets.iter().copied().chain(None),
                SwitchInt { ref targets, .. } => targets.targets.iter().copied().chain(None),
                FalseEdge { ref real_target, imaginary_target } => {
                    slice::from_ref(real_target).into_iter().copied().chain(Some(imaginary_target))
                }
            }
        }

        #[inline]
        #[define_opaque(SuccessorsMut)]
        pub fn successors_mut(&mut self) -> SuccessorsMut<'_> {
            use self::TerminatorKind::*;
            match *self {
                Call {
                    target: Some(ref mut t), unwind: UnwindAction::Cleanup(ref mut u), ..
                }
                | Yield { resume: ref mut t, drop: Some(ref mut u), .. }
                | Drop { target: ref mut t, unwind: UnwindAction::Cleanup(ref mut u), .. }
                | Assert { target: ref mut t, unwind: UnwindAction::Cleanup(ref mut u), .. }
                | FalseUnwind {
                    real_target: ref mut t,
                    unwind: UnwindAction::Cleanup(ref mut u),
                } => slice::from_mut(t).into_iter().chain(Some(u)),
                Goto { target: ref mut t }
                | Call { target: None, unwind: UnwindAction::Cleanup(ref mut t), .. }
                | Call { target: Some(ref mut t), unwind: _, .. }
                | Yield { resume: ref mut t, drop: None, .. }
                | Drop { target: ref mut t, unwind: _, .. }
                | Assert { target: ref mut t, unwind: _, .. }
                | FalseUnwind { real_target: ref mut t, unwind: _ } => {
                    slice::from_mut(t).into_iter().chain(None)
                }
                UnwindResume
                | UnwindTerminate(_)
                | CoroutineDrop
                | Return
                | Unreachable
                | TailCall { .. }
                | Call { target: None, unwind: _, .. } => (&mut []).into_iter().chain(None),
                InlineAsm { ref mut targets, unwind: UnwindAction::Cleanup(ref mut u), .. } => {
                    targets.iter_mut().chain(Some(u))
                }
                InlineAsm { ref mut targets, unwind: _, .. } => targets.iter_mut().chain(None),
                SwitchInt { ref mut targets, .. } => targets.targets.iter_mut().chain(None),
                FalseEdge { ref mut real_target, ref mut imaginary_target } => {
                    slice::from_mut(real_target).into_iter().chain(Some(imaginary_target))
                }
            }
        }
    }
}

impl<'tcx> TerminatorKind<'tcx> {
    #[inline]
    pub fn unwind(&self) -> Option<&UnwindAction> {
        match *self {
            TerminatorKind::Goto { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Return
            | TerminatorKind::TailCall { .. }
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

    #[inline]
    pub fn unwind_mut(&mut self) -> Option<&mut UnwindAction> {
        match *self {
            TerminatorKind::Goto { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Return
            | TerminatorKind::TailCall { .. }
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

    #[inline]
    pub fn as_switch(&self) -> Option<(&Operand<'tcx>, &SwitchTargets)> {
        match self {
            TerminatorKind::SwitchInt { discr, targets } => Some((discr, targets)),
            _ => None,
        }
    }

    #[inline]
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
    /// For terminators that have a single successor, like `goto`, and `assert` without a cleanup
    /// block.
    Single(BasicBlock),
    /// For terminators that have two successors, like `assert` with a cleanup block, and
    /// `falseEdge`.
    Double(BasicBlock, BasicBlock),
    /// Special action for `Yield`, `Call` and `InlineAsm` terminators.
    AssignOnReturn {
        return_: &'mir [BasicBlock],
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
            Return
            | TailCall { .. }
            | UnwindResume
            | UnwindTerminate(_)
            | CoroutineDrop
            | Unreachable => TerminatorEdges::None,

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

            Yield { resume: ref target, drop, resume_arg, value: _ } => {
                TerminatorEdges::AssignOnReturn {
                    return_: slice::from_ref(target),
                    cleanup: drop,
                    place: CallReturnPlaces::Yield(resume_arg),
                }
            }

            Call {
                unwind,
                destination,
                ref target,
                func: _,
                args: _,
                fn_span: _,
                call_source: _,
            } => TerminatorEdges::AssignOnReturn {
                return_: target.as_ref().map(slice::from_ref).unwrap_or_default(),
                cleanup: unwind.cleanup_block(),
                place: CallReturnPlaces::Call(destination),
            },

            InlineAsm {
                asm_macro: _,
                template: _,
                ref operands,
                options: _,
                line_spans: _,
                ref targets,
                unwind,
            } => TerminatorEdges::AssignOnReturn {
                return_: targets,
                cleanup: unwind.cleanup_block(),
                place: CallReturnPlaces::InlineAsm(operands),
            },

            SwitchInt { ref targets, ref discr } => TerminatorEdges::SwitchInt { targets, discr },
        }
    }
}

impl CallSource {
    pub fn from_hir_call(self) -> bool {
        matches!(self, CallSource::Normal)
    }
}

impl InlineAsmMacro {
    pub const fn diverges(self, options: InlineAsmOptions) -> bool {
        match self {
            InlineAsmMacro::Asm => options.contains(InlineAsmOptions::NORETURN),
            InlineAsmMacro::NakedAsm => true,
        }
    }
}
