//! Validates the MIR to ensure that invariants are upheld.

use super::{MirPass, MirSource};
use rustc_hir::lang_items::FnOnceTraitLangItem;
use rustc_hir::Constness;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::{
    mir::{
        BasicBlock, Body, Location, Operand, Rvalue, Statement, StatementKind, Terminator,
        TerminatorKind,
    },
    ty::{self, ParamEnv, ToPredicate, Ty, TyCtxt},
};
use rustc_span::def_id::DefId;
use rustc_trait_selection::traits;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;

#[derive(Copy, Clone, Debug)]
enum EdgeKind {
    Unwind,
    Normal,
}

pub struct Validator {
    /// Describes at which point in the pipeline this validation is happening.
    pub when: String,
}

impl<'tcx> MirPass<'tcx> for Validator {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        let def_id = source.def_id();
        let param_env = tcx.param_env(def_id);
        TypeChecker { when: &self.when, def_id, body, tcx, param_env }.visit_body(body);
    }
}

struct TypeChecker<'a, 'tcx> {
    when: &'a str,
    def_id: DefId,
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
}

impl<'a, 'tcx> TypeChecker<'a, 'tcx> {
    fn fail(&self, location: Location, msg: impl AsRef<str>) {
        let span = self.body.source_info(location).span;
        // We use `delay_span_bug` as we might see broken MIR when other errors have already
        // occurred.
        self.tcx.sess.diagnostic().delay_span_bug(
            span,
            &format!(
                "broken MIR in {:?} ({}) at {:?}:\n{}",
                self.def_id,
                self.when,
                location,
                msg.as_ref()
            ),
        );
    }

    fn check_edge(&self, location: Location, bb: BasicBlock, edge_kind: EdgeKind) {
        if let Some(bb) = self.body.basic_blocks().get(bb) {
            let src = self.body.basic_blocks().get(location.block).unwrap();
            match (src.is_cleanup, bb.is_cleanup, edge_kind) {
                // Non-cleanup blocks can jump to non-cleanup blocks along non-unwind edges
                (false, false, EdgeKind::Normal)
                // Non-cleanup blocks can jump to cleanup blocks along unwind edges
                | (false, true, EdgeKind::Unwind)
                // Cleanup blocks can jump to cleanup blocks along non-unwind edges
                | (true, true, EdgeKind::Normal) => {}
                // All other jumps are invalid
                _ => {
                    self.fail(
                        location,
                        format!(
                            "{:?} edge to {:?} violates unwind invariants (cleanup {:?} -> {:?})",
                            edge_kind,
                            bb,
                            src.is_cleanup,
                            bb.is_cleanup,
                        )
                    )
                }
            }
        } else {
            self.fail(location, format!("encountered jump to invalid basic block {:?}", bb))
        }
    }

    fn check_ty_callable(&self, location: Location, ty: Ty<'tcx>) {
        if let ty::FnPtr(..) | ty::FnDef(..) = ty.kind {
            // We have a `FnPtr` or `FnDef` which is trivially safe to call.
        } else {
            // FIXME(doctorn): call shims shouldn't reference unnormalized types, so
            // this case should be removed.
            // We haven't got a `FnPtr` or `FnDef` but we are still safe to call it if it
            // implements `FnOnce` (as `Fn: FnMut` and `FnMut: FnOnce`).
            let fn_once_trait = self.tcx.require_lang_item(FnOnceTraitLangItem, None);
            let item_def_id = self
                .tcx
                .associated_items(fn_once_trait)
                .in_definition_order()
                .next()
                .unwrap()
                .def_id;
            self.tcx.infer_ctxt().enter(|infcx| {
                let trait_ref = ty::TraitRef {
                    def_id: fn_once_trait,
                    substs: self.tcx.mk_substs_trait(
                        ty,
                        infcx.fresh_substs_for_item(self.body.span, item_def_id),
                    ),
                };
                let predicate = ty::PredicateKind::Trait(
                    ty::Binder::bind(ty::TraitPredicate { trait_ref }),
                    Constness::NotConst,
                )
                .to_predicate(self.tcx);
                let obligation = traits::Obligation::new(
                    traits::ObligationCause::dummy(),
                    self.param_env,
                    predicate,
                );
                if !infcx.predicate_may_hold(&obligation) {
                    self.fail(
                        location,
                        format!("encountered non-callable type {} in `Call` terminator", ty),
                    );
                }
            });
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for TypeChecker<'a, 'tcx> {
    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        // `Operand::Copy` is only supposed to be used with `Copy` types.
        if let Operand::Copy(place) = operand {
            let ty = place.ty(&self.body.local_decls, self.tcx).ty;
            let span = self.body.source_info(location).span;

            if !ty.is_copy_modulo_regions(self.tcx, self.param_env, span) {
                self.fail(location, format!("`Operand::Copy` with non-`Copy` type {}", ty));
            }
        }

        self.super_operand(operand, location);
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        // The sides of an assignment must not alias. Currently this just checks whether the places
        // are identical.
        if let StatementKind::Assign(box (dest, rvalue)) = &statement.kind {
            match rvalue {
                Rvalue::Use(Operand::Copy(src) | Operand::Move(src)) => {
                    if dest == src {
                        self.fail(
                            location,
                            "encountered `Assign` statement with overlapping memory",
                        );
                    }
                }
                _ => {}
            }
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        match &terminator.kind {
            TerminatorKind::Goto { target } => {
                self.check_edge(location, *target, EdgeKind::Normal);
            }
            TerminatorKind::SwitchInt { targets, values, .. } => {
                if targets.len() != values.len() + 1 {
                    self.fail(
                        location,
                        format!(
                            "encountered `SwitchInt` terminator with {} values, but {} targets (should be values+1)",
                            values.len(),
                            targets.len(),
                        ),
                    );
                }
                for target in targets {
                    self.check_edge(location, *target, EdgeKind::Normal);
                }
            }
            TerminatorKind::Drop { target, unwind, .. } => {
                self.check_edge(location, *target, EdgeKind::Normal);
                if let Some(unwind) = unwind {
                    self.check_edge(location, *unwind, EdgeKind::Unwind);
                }
            }
            TerminatorKind::DropAndReplace { target, unwind, .. } => {
                self.check_edge(location, *target, EdgeKind::Normal);
                if let Some(unwind) = unwind {
                    self.check_edge(location, *unwind, EdgeKind::Unwind);
                }
            }
            TerminatorKind::Call { func, destination, cleanup, .. } => {
                self.check_ty_callable(location, &func.ty(&self.body.local_decls, self.tcx));
                if let Some((_, target)) = destination {
                    self.check_edge(location, *target, EdgeKind::Normal);
                }
                if let Some(cleanup) = cleanup {
                    self.check_edge(location, *cleanup, EdgeKind::Unwind);
                }
            }
            TerminatorKind::Assert { cond, target, cleanup, .. } => {
                let cond_ty = cond.ty(&self.body.local_decls, self.tcx);
                if cond_ty != self.tcx.types.bool {
                    self.fail(
                        location,
                        format!(
                            "encountered non-boolean condition of type {} in `Assert` terminator",
                            cond_ty
                        ),
                    );
                }
                self.check_edge(location, *target, EdgeKind::Normal);
                if let Some(cleanup) = cleanup {
                    self.check_edge(location, *cleanup, EdgeKind::Unwind);
                }
            }
            TerminatorKind::Yield { resume, drop, .. } => {
                self.check_edge(location, *resume, EdgeKind::Normal);
                if let Some(drop) = drop {
                    self.check_edge(location, *drop, EdgeKind::Normal);
                }
            }
            TerminatorKind::FalseEdge { real_target, imaginary_target } => {
                self.check_edge(location, *real_target, EdgeKind::Normal);
                self.check_edge(location, *imaginary_target, EdgeKind::Normal);
            }
            TerminatorKind::FalseUnwind { real_target, unwind } => {
                self.check_edge(location, *real_target, EdgeKind::Normal);
                if let Some(unwind) = unwind {
                    self.check_edge(location, *unwind, EdgeKind::Unwind);
                }
            }
            TerminatorKind::InlineAsm { destination, .. } => {
                if let Some(destination) = destination {
                    self.check_edge(location, *destination, EdgeKind::Normal);
                }
            }
            // Nothing to validate for these.
            TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::GeneratorDrop => {}
        }
    }
}
