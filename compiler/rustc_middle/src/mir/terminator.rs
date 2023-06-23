use smallvec::SmallVec;

use super::{BasicBlock, InlineAsmOperand, Operand, SourceInfo, TerminatorKind, UnwindAction};
use rustc_ast::InlineAsmTemplatePiece;
pub use rustc_ast::Mutability;
use rustc_macros::HashStable;
use std::borrow::Cow;
use std::fmt::{self, Debug, Formatter, Write};
use std::iter;
use std::slice;

pub use super::query::*;

#[derive(Debug, Clone, TyEncodable, TyDecodable, Hash, HashStable, PartialEq)]
pub struct SwitchTargets {
    /// Possible values. The locations to branch to in each case
    /// are found in the corresponding indices from the `targets` vector.
    values: SmallVec<[u128; 1]>,

    /// Possible branch sites. The last element of this vector is used
    /// for the otherwise branch, so targets.len() == values.len() + 1
    /// should hold.
    //
    // This invariant is quite non-obvious and also could be improved.
    // One way to make this invariant is to have something like this instead:
    //
    // branches: Vec<(ConstInt, BasicBlock)>,
    // otherwise: Option<BasicBlock> // exhaustive if None
    //
    // However we’ve decided to keep this as-is until we figure a case
    // where some other approach seems to be strictly better than other.
    targets: SmallVec<[BasicBlock; 2]>,
}

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
            Resume
            | Terminate
            | GeneratorDrop
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
            Resume
            | Terminate
            | GeneratorDrop
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
            | TerminatorKind::Resume
            | TerminatorKind::Terminate
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::GeneratorDrop
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
            | TerminatorKind::Resume
            | TerminatorKind::Terminate
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::GeneratorDrop
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

impl<'tcx> Debug for TerminatorKind<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_head(fmt)?;
        let successor_count = self.successors().count();
        let labels = self.fmt_successor_labels();
        assert_eq!(successor_count, labels.len());

        let unwind = match self.unwind() {
            // Not needed or included in successors
            None | Some(UnwindAction::Cleanup(_)) => None,
            Some(UnwindAction::Continue) => Some("unwind continue"),
            Some(UnwindAction::Unreachable) => Some("unwind unreachable"),
            Some(UnwindAction::Terminate) => Some("unwind terminate"),
        };

        match (successor_count, unwind) {
            (0, None) => Ok(()),
            (0, Some(unwind)) => write!(fmt, " -> {}", unwind),
            (1, None) => write!(fmt, " -> {:?}", self.successors().next().unwrap()),
            _ => {
                write!(fmt, " -> [")?;
                for (i, target) in self.successors().enumerate() {
                    if i > 0 {
                        write!(fmt, ", ")?;
                    }
                    write!(fmt, "{}: {:?}", labels[i], target)?;
                }
                if let Some(unwind) = unwind {
                    write!(fmt, ", {unwind}")?;
                }
                write!(fmt, "]")
            }
        }
    }
}

impl<'tcx> TerminatorKind<'tcx> {
    /// Writes the "head" part of the terminator; that is, its name and the data it uses to pick the
    /// successor basic block, if any. The only information not included is the list of possible
    /// successors, which may be rendered differently between the text and the graphviz format.
    pub fn fmt_head<W: Write>(&self, fmt: &mut W) -> fmt::Result {
        use self::TerminatorKind::*;
        match self {
            Goto { .. } => write!(fmt, "goto"),
            SwitchInt { discr, .. } => write!(fmt, "switchInt({:?})", discr),
            Return => write!(fmt, "return"),
            GeneratorDrop => write!(fmt, "generator_drop"),
            Resume => write!(fmt, "resume"),
            Terminate => write!(fmt, "abort"),
            Yield { value, resume_arg, .. } => write!(fmt, "{:?} = yield({:?})", resume_arg, value),
            Unreachable => write!(fmt, "unreachable"),
            Drop { place, .. } => write!(fmt, "drop({:?})", place),
            Call { func, args, destination, .. } => {
                write!(fmt, "{:?} = ", destination)?;
                write!(fmt, "{:?}(", func)?;
                for (index, arg) in args.iter().enumerate() {
                    if index > 0 {
                        write!(fmt, ", ")?;
                    }
                    write!(fmt, "{:?}", arg)?;
                }
                write!(fmt, ")")
            }
            Assert { cond, expected, msg, .. } => {
                write!(fmt, "assert(")?;
                if !expected {
                    write!(fmt, "!")?;
                }
                write!(fmt, "{:?}, ", cond)?;
                msg.fmt_assert_args(fmt)?;
                write!(fmt, ")")
            }
            FalseEdge { .. } => write!(fmt, "falseEdge"),
            FalseUnwind { .. } => write!(fmt, "falseUnwind"),
            InlineAsm { template, ref operands, options, .. } => {
                write!(fmt, "asm!(\"{}\"", InlineAsmTemplatePiece::to_string(template))?;
                for op in operands {
                    write!(fmt, ", ")?;
                    let print_late = |&late| if late { "late" } else { "" };
                    match op {
                        InlineAsmOperand::In { reg, value } => {
                            write!(fmt, "in({}) {:?}", reg, value)?;
                        }
                        InlineAsmOperand::Out { reg, late, place: Some(place) } => {
                            write!(fmt, "{}out({}) {:?}", print_late(late), reg, place)?;
                        }
                        InlineAsmOperand::Out { reg, late, place: None } => {
                            write!(fmt, "{}out({}) _", print_late(late), reg)?;
                        }
                        InlineAsmOperand::InOut {
                            reg,
                            late,
                            in_value,
                            out_place: Some(out_place),
                        } => {
                            write!(
                                fmt,
                                "in{}out({}) {:?} => {:?}",
                                print_late(late),
                                reg,
                                in_value,
                                out_place
                            )?;
                        }
                        InlineAsmOperand::InOut { reg, late, in_value, out_place: None } => {
                            write!(fmt, "in{}out({}) {:?} => _", print_late(late), reg, in_value)?;
                        }
                        InlineAsmOperand::Const { value } => {
                            write!(fmt, "const {:?}", value)?;
                        }
                        InlineAsmOperand::SymFn { value } => {
                            write!(fmt, "sym_fn {:?}", value)?;
                        }
                        InlineAsmOperand::SymStatic { def_id } => {
                            write!(fmt, "sym_static {:?}", def_id)?;
                        }
                    }
                }
                write!(fmt, ", options({:?}))", options)
            }
        }
    }

    /// Returns the list of labels for the edges to the successor basic blocks.
    pub fn fmt_successor_labels(&self) -> Vec<Cow<'static, str>> {
        use self::TerminatorKind::*;
        match *self {
            Return | Resume | Terminate | Unreachable | GeneratorDrop => vec![],
            Goto { .. } => vec!["".into()],
            SwitchInt { ref targets, .. } => targets
                .values
                .iter()
                .map(|&u| Cow::Owned(u.to_string()))
                .chain(iter::once("otherwise".into()))
                .collect(),
            Call { target: Some(_), unwind: UnwindAction::Cleanup(_), .. } => {
                vec!["return".into(), "unwind".into()]
            }
            Call { target: Some(_), unwind: _, .. } => vec!["return".into()],
            Call { target: None, unwind: UnwindAction::Cleanup(_), .. } => vec!["unwind".into()],
            Call { target: None, unwind: _, .. } => vec![],
            Yield { drop: Some(_), .. } => vec!["resume".into(), "drop".into()],
            Yield { drop: None, .. } => vec!["resume".into()],
            Drop { unwind: UnwindAction::Cleanup(_), .. } => vec!["return".into(), "unwind".into()],
            Drop { unwind: _, .. } => vec!["return".into()],
            Assert { unwind: UnwindAction::Cleanup(_), .. } => {
                vec!["success".into(), "unwind".into()]
            }
            Assert { unwind: _, .. } => vec!["success".into()],
            FalseEdge { .. } => vec!["real".into(), "imaginary".into()],
            FalseUnwind { unwind: UnwindAction::Cleanup(_), .. } => {
                vec!["real".into(), "unwind".into()]
            }
            FalseUnwind { unwind: _, .. } => vec!["real".into()],
            InlineAsm { destination: Some(_), unwind: UnwindAction::Cleanup(_), .. } => {
                vec!["return".into(), "unwind".into()]
            }
            InlineAsm { destination: Some(_), unwind: _, .. } => {
                vec!["return".into()]
            }
            InlineAsm { destination: None, unwind: UnwindAction::Cleanup(_), .. } => {
                vec!["unwind".into()]
            }
            InlineAsm { destination: None, unwind: _, .. } => vec![],
        }
    }
}
