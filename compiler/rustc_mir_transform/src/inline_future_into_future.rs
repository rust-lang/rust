//! Converts `y = <Future as IntoFuture>::into_future(x);` into just `y = x;`,
//! since we "know" that matches the behavior of the blanket implementation of
//! IntoFuture for F where F: Future.
//!
//! FIXME: determine such coalescing is sound. In particular, check whether
//! specialization could foil our plans here!
//!
//! This is meant to enhance the effectiveness of the upvar-to-local-prop
//! transformation in reducing the size of the generators constructed by the
//! compiler.

use crate::MirPass;
use rustc_index::IndexVec;
use rustc_middle::mir::interpret::ConstValue;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::DefId;

pub struct InlineFutureIntoFuture;
impl<'tcx> MirPass<'tcx> for InlineFutureIntoFuture {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0 // on by default w/o -Zmir-opt-level=0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let Some(into_future_fn_def_id) = tcx.lang_items().into_future_fn() else { return; };
        let Some(future_trait_def_id) = tcx.lang_items().future_trait() else { return; };
        let mir_source_def_id = body.source.def_id();
        trace!("Running InlineFutureIntoFuture on {:?}", body.source);
        let local_decls = body.local_decls().to_owned();
        let mut v = Inliner {
            tcx,
            into_future_fn_def_id,
            future_trait_def_id,
            mir_source_def_id,
            local_decls,
        };
        v.visit_body(body);
    }
}

struct Inliner<'tcx> {
    tcx: TyCtxt<'tcx>,
    mir_source_def_id: DefId,
    into_future_fn_def_id: DefId,
    future_trait_def_id: DefId,
    local_decls: IndexVec<Local, LocalDecl<'tcx>>,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum FoundImplFuture {
    Yes,
    No,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum FoundIntoFutureCall {
    Yes,
    No,
}

struct ImplFutureCallingIntoFuture<'tcx> {
    args: Vec<Operand<'tcx>>,
    destination: Place<'tcx>,
    target: Option<BasicBlock>,
}

impl<'tcx> Inliner<'tcx> {
    // This verifies that `ty` implements `Future`, according to the where
    // clauses (i.e. predicates) attached to the source code identified by
    // `mir_source_def_id`).
    fn does_ty_impl_future(&self, ty: Ty<'tcx>) -> FoundImplFuture {
        let mir_source_predicates = self.tcx.predicates_of(self.mir_source_def_id);
        let predicates = mir_source_predicates.instantiate_identity(self.tcx);
        for pred in &predicates.predicates {
            let Some(kind) = pred.kind().no_bound_vars() else { continue; };
            let ty::ClauseKind::Trait(trait_pred) = kind else { continue; };
            let ty::TraitPredicate { trait_ref, polarity: ty::ImplPolarity::Positive } = trait_pred else { continue };

            // FIXME: justify ignoring `substs` below. My current argument is
            // that `trait Future` has no generic parameters, and the blanket
            // impl of `IntoFuture` for all futures does not put any constrants
            // on the associated items of those futures. But it is worth running
            // this by a trait system expert to validate.
            let ty::TraitRef { def_id: trait_def_id, .. } = trait_ref;
            let self_ty = trait_ref.self_ty();
            if trait_def_id == self.future_trait_def_id {
                if self_ty == ty {
                    return FoundImplFuture::Yes;
                }
            }
        }
        FoundImplFuture::No
    }
}

impl<'tcx> MutVisitor<'tcx> for Inliner<'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'tcx> {
        self.tcx
    }
    fn visit_basic_block_data(&mut self, _bb: BasicBlock, bb_data: &mut BasicBlockData<'tcx>) {
        let Some(term) = &mut bb_data.terminator else { return; };
        let Some(result) = self.analyze_terminator(term) else { return; };
        let ImplFutureCallingIntoFuture {
            args, destination: dest, target: Some(target)
        } = result else { return; };

        // At this point, we have identified this terminator as a call to the
        // associated function `<impl Future as IntoFuture>::into_future`
        // Due to our knowledge of how libcore implements Future and IntoFuture,
        // we know we can replace such a call with a trivial move.

        let Some(arg0) = args.get(0) else { return; };

        trace!("InlineFutureIntoFuture bb_data args:{args:?} dest:{dest:?} target:{target:?}");

        bb_data.statements.push(Statement {
            source_info: term.source_info,
            kind: StatementKind::Assign(Box::new((dest, Rvalue::Use(arg0.clone())))),
        });
        term.kind = TerminatorKind::Goto { target }
    }
}

impl<'tcx> Inliner<'tcx> {
    fn analyze_terminator(
        &mut self,
        term: &mut Terminator<'tcx>,
    ) -> Option<ImplFutureCallingIntoFuture<'tcx>> {
        let mut found = (FoundImplFuture::No, FoundIntoFutureCall::No);
        let &TerminatorKind::Call {
            ref func, ref args, destination, target, fn_span: _, unwind: _, call_source: _
        } = &term.kind else { return None; };
        let Operand::Constant(c) = func else { return None; };
        let ConstantKind::Val(val_const, const_ty) = c.literal else { return None; };
        let ConstValue::ZeroSized = val_const else { return None; };
        let ty::FnDef(fn_def_id, substs) =  const_ty.kind() else { return None; };
        if *fn_def_id == self.into_future_fn_def_id {
            found.1 = FoundIntoFutureCall::Yes;
        } else {
            trace!("InlineFutureIntoFuture bail as this is not `into_future` invocation.");
            return None;
        }
        let arg0_ty = args.get(0).map(|arg0| arg0.ty(&self.local_decls, self.tcx()));
        trace!("InlineFutureIntoFuture substs:{substs:?} args:{args:?} arg0 ty:{arg0_ty:?}");
        let Some(arg0_ty) = arg0_ty else { return None; };
        found.0 = self.does_ty_impl_future(arg0_ty);
        if let (FoundImplFuture::Yes, FoundIntoFutureCall::Yes) = found {
            trace!("InlineFutureIntoFuture can replace {term:?}, a {func:?} call, with move");
            if !self.tcx.consider_optimizing(|| {
                format!("InlineFutureIntoFuture {:?}", self.mir_source_def_id)
            }) {
                return None;
            }
            let args = args.clone();
            Some(ImplFutureCallingIntoFuture { args, destination, target })
        } else {
            None
        }
    }
}
