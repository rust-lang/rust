use crate::borrow_pats::info::VarInfo;
use crate::borrow_pats::{
    construct_visit_order, unloop_preds, BodyStats, DropKind, PlaceMagic, PrintPrevent, SimpleTyKind, VisitKind,
};

use super::super::{calc_call_local_relations, CfgInfo, DataInfo, LocalInfo, LocalOrConst};
use super::LocalKind;

use clippy_utils::ty::{has_drop, is_copy};
use mid::mir::visit::Visitor;
use mid::mir::{Body, Terminator};
use mid::ty::TyCtxt;
use rustc_data_structures::fx::FxHashMap;
use rustc_index::IndexVec;
use rustc_lint::LateContext;
use rustc_middle as mid;
use rustc_middle::mir;
use rustc_middle::mir::{BasicBlock, Local, Place, Rvalue};
use smallvec::SmallVec;

/// This analysis is special as it is always the first one to run. It collects
/// information about the control flow etc, which will be used by future analysis.
///
/// For better construction and value tracking, it uses reverse order depth search
#[derive(Debug)]
pub struct MetaAnalysis<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: PrintPrevent<TyCtxt<'tcx>>,
    cx: PrintPrevent<&'tcx LateContext<'tcx>>,
    pub cfg: IndexVec<BasicBlock, CfgInfo>,
    pub terms: FxHashMap<BasicBlock, FxHashMap<Local, Vec<Local>>>,
    pub return_block: BasicBlock,
    pub locals: IndexVec<Local, LocalInfo<'tcx>>,
    pub preds: IndexVec<BasicBlock, SmallVec<[BasicBlock; 1]>>,
    pub preds_unlooped: IndexVec<BasicBlock, SmallVec<[BasicBlock; 1]>>,
    pub visit_order: Vec<VisitKind>,
    pub stats: BodyStats,
}

impl<'a, 'tcx> MetaAnalysis<'a, 'tcx> {
    pub fn from_body(cx: &'tcx LateContext<'tcx>, body: &'a Body<'tcx>) -> Self {
        let mut anly = Self::new(cx, body);
        anly.visit_body(body);
        anly.unloop_preds();
        anly.visit_order = construct_visit_order(body, &anly.cfg, &anly.preds_unlooped);

        anly
    }

    pub fn new(cx: &'tcx LateContext<'tcx>, body: &'a Body<'tcx>) -> Self {
        let locals = Self::setup_local_infos(body);
        let bb_len = body.basic_blocks.len();

        let mut preds = IndexVec::with_capacity(bb_len);
        preds.resize(bb_len, SmallVec::new());

        let mut cfg = IndexVec::with_capacity(bb_len);
        cfg.resize(bb_len, CfgInfo::None);

        Self {
            body,
            tcx: PrintPrevent(cx.tcx),
            cx: PrintPrevent(cx),
            cfg,
            terms: Default::default(),
            return_block: BasicBlock::from_u32(0),
            locals,
            preds,
            preds_unlooped: IndexVec::with_capacity(bb_len),
            visit_order: Default::default(),
            stats: Default::default(),
        }
    }

    fn setup_local_infos(body: &mir::Body<'tcx>) -> IndexVec<Local, LocalInfo<'tcx>> {
        let local_info_iter = body.local_decls.indices().map(|_| LocalInfo::new(LocalKind::AnonVar));
        let mut local_infos = IndexVec::new();
        local_infos.extend(local_info_iter);

        local_infos[super::super::RETURN].kind = LocalKind::Return;

        local_infos
    }

    fn unloop_preds(&mut self) {
        self.preds_unlooped = unloop_preds(&self.cfg, &self.preds);
    }

    fn visit_terminator_for_cfg(&mut self, term: &Terminator<'tcx>, bb: BasicBlock) {
        let cfg_info = match &term.kind {
            #[rustfmt::skip]
            mir::TerminatorKind::FalseEdge { real_target: target, .. }
            | mir::TerminatorKind::FalseUnwind { real_target: target, .. }
            | mir::TerminatorKind::Assert { target, .. }
            | mir::TerminatorKind::Call { target: Some(target), .. }
            | mir::TerminatorKind::Drop { target, .. }
            | mir::TerminatorKind::InlineAsm { destination: Some(target), .. }
            | mir::TerminatorKind::Goto { target } => {
                self.preds[*target].push(bb);
                CfgInfo::Linear(*target)
            },
            mir::TerminatorKind::SwitchInt { targets, .. } => {
                let mut branches = SmallVec::new();
                branches.extend_from_slice(targets.all_targets());

                for target in &branches {
                    self.preds[*target].push(bb);
                }

                CfgInfo::Condition { branches }
            },
            #[rustfmt::skip]
            mir::TerminatorKind::UnwindResume
            | mir::TerminatorKind::UnwindTerminate(_)
            | mir::TerminatorKind::Unreachable
            | mir::TerminatorKind::CoroutineDrop
            | mir::TerminatorKind::Call { .. }
            | mir::TerminatorKind::InlineAsm { .. } => {
                CfgInfo::None
            },
            mir::TerminatorKind::Return => {
                self.return_block = bb;
                CfgInfo::Return
            },
            mir::TerminatorKind::Yield { .. } => unreachable!(),
        };

        self.cfg[bb] = cfg_info;
    }

    fn visit_terminator_for_terms(&mut self, term: &Terminator<'tcx>, bb: BasicBlock) {
        if let 
            mir::TerminatorKind::Call {
                func,
                args,
                destination,
                ..
            } = &term.kind {
                assert!(destination.projection.is_empty());
                let dest = destination.local;
                self.terms.insert(
                    bb,
                    calc_call_local_relations(self.tcx.0, self.body, func, dest, args, &mut self.stats),
                );
            }
    }

    fn visit_terminator_for_locals(&mut self, term: &Terminator<'tcx>, _bb: BasicBlock) {
        if let mir::TerminatorKind::Call { destination, .. } = &term.kind {
                // TODO: Should mut arguments be handled?
                assert!(destination.projection.is_empty());
                let local = destination.local;
                self.locals
                    .get_mut(local)
                    .unwrap()
                    .add_assign(*destination, DataInfo::Computed);
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for MetaAnalysis<'a, 'tcx> {
    fn visit_var_debug_info(&mut self, info: &mir::VarDebugInfo<'tcx>) {
        if let mir::VarDebugInfoContents::Place(place) = info.value {
            assert!(place.just_local());
            let local = place.local;
            if let Some(local_info) = self.locals.get_mut(local) {
                let decl = &self.body.local_decls[local];
                let drop = if !decl.ty.needs_drop(self.tcx.0, self.cx.0.param_env) {
                    DropKind::NonDrop
                // } else if decl.ty.has_significant_drop(self.tcx.0, self.cx.0.param_env) {
                } else if has_drop(self.cx.0, decl.ty) {
                    DropKind::SelfDrop
                } else {
                    DropKind::PartDrop
                };
                let var_info = VarInfo {
                    argument: info.argument_index.is_some(),
                    mutable: decl.mutability.is_mut(),
                    owned: !decl.ty.is_ref(),
                    copy: is_copy(self.cx.0, decl.ty),
                    // Turns out that both `has_significant_drop` and `has_drop`
                    // return false if only fields require drops. Strings are a
                    // good testing example for this.
                    drop,
                    ty: SimpleTyKind::from_ty(decl.ty),
                };

                local_info.kind = LocalKind::UserVar(info.name, var_info);

                if local_info.kind.is_arg() {
                    // +1 since it's assigned outside of the body
                    local_info.assign_count += 1;
                    local_info.add_assign(place, DataInfo::Argument);
                }
            }
        } else {
            todo!("How should this be handled? {info:#?}");
        }
    }

    fn visit_terminator(&mut self, term: &Terminator<'tcx>, loc: mir::Location) {
        self.visit_terminator_for_cfg(term, loc.block);
        self.visit_terminator_for_terms(term, loc.block);
        self.visit_terminator_for_locals(term, loc.block);
    }

    fn visit_assign(&mut self, place: &Place<'tcx>, rval: &Rvalue<'tcx>, _loc: mir::Location) {
        let local = place.local;

        let assign_info = match rval {
            mir::Rvalue::Ref(_reg, _kind, src) => {
                match src.projection.as_slice() {
                    [mir::PlaceElem::Deref] => {
                        // &(*_1) = Copy
                        DataInfo::Local(src.local)
                    },
                    _ => DataInfo::Loan(*src),
                }
            },
            mir::Rvalue::Use(op) => match &op {
                mir::Operand::Copy(other) | mir::Operand::Move(other) => {
                    if other.is_part() {
                        DataInfo::Part(*other)
                    } else {
                        DataInfo::Local(other.local)
                    }
                },
                mir::Operand::Constant(_) => DataInfo::Const,
            },

            // Constructed Values
            Rvalue::Aggregate(_, fields) => {
                let parts = fields.iter().map(LocalOrConst::from).collect();
                DataInfo::Ctor(parts)
            },
            Rvalue::Repeat(op, _) => DataInfo::Ctor(vec![op.into()]),

            // Casts should depend on the input data
            Rvalue::Cast(_kind, op, _target) => {
                if let Some(place) = op.place() {
                    assert!(place.just_local());
                    DataInfo::Cast(place.local)
                } else {
                    DataInfo::Const
                }
            },

            Rvalue::NullaryOp(_, _) => DataInfo::Const,

            Rvalue::ThreadLocalRef(_)
            | Rvalue::AddressOf(_, _)
            | Rvalue::Len(_)
            | Rvalue::BinaryOp(_, _)
            | Rvalue::CheckedBinaryOp(_, _)
            | Rvalue::UnaryOp(_, _)
            | Rvalue::Discriminant(_)
            | Rvalue::ShallowInitBox(_, _)
            | Rvalue::CopyForDeref(_) => DataInfo::Computed,
        };

        self.locals.get_mut(local).unwrap().add_assign(*place, assign_info);
    }
}
