use rustc_data_structures::graph::iterate::{
    NodeStatus, TriColorDepthFirstSearch, TriColorVisitor,
};
use rustc_hir::intravisit::FnKind;
use rustc_middle::mir::{BasicBlock, Body, Operand, TerminatorKind};
use rustc_middle::ty::subst::{GenericArg, InternalSubsts};
use rustc_middle::ty::{self, AssocItem, AssocItemContainer, Instance, TyCtxt};
use rustc_session::lint::builtin::UNCONDITIONAL_RECURSION;
use rustc_span::Span;
use std::ops::ControlFlow;

crate fn check<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
    let def_id = body.source.def_id().expect_local();

    if let Some(fn_kind) = tcx.hir().get_by_def_id(def_id).fn_kind() {
        if let FnKind::Closure = fn_kind {
            // closures can't recur, so they don't matter.
            return;
        }

        // If this is trait/impl method, extract the trait's substs.
        let trait_substs = match tcx.opt_associated_item(def_id.to_def_id()) {
            Some(AssocItem {
                container: AssocItemContainer::TraitContainer(trait_def_id), ..
            }) => {
                let trait_substs_count = tcx.generics_of(*trait_def_id).count();
                &InternalSubsts::identity_for_item(tcx, def_id.to_def_id())[..trait_substs_count]
            }
            _ => &[],
        };

        let mut vis = Search { tcx, body, reachable_recursive_calls: vec![], trait_substs };
        if let Some(NonRecursive) = TriColorDepthFirstSearch::new(&body).run_from_start(&mut vis) {
            return;
        }
        if vis.reachable_recursive_calls.is_empty() {
            return;
        }

        vis.reachable_recursive_calls.sort();

        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
        let sp = tcx.sess.source_map().guess_head_span(tcx.hir().span_with_body(hir_id));
        tcx.struct_span_lint_hir(UNCONDITIONAL_RECURSION, hir_id, sp, |lint| {
            let mut db = lint.build("function cannot return without recursing");
            db.span_label(sp, "cannot return without recursing");
            // offer some help to the programmer.
            for call_span in vis.reachable_recursive_calls {
                db.span_label(call_span, "recursive call site");
            }
            db.help("a `loop` may express intention better if this is on purpose");
            db.emit();
        });
    }
}

struct NonRecursive;

struct Search<'mir, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'mir Body<'tcx>,
    trait_substs: &'tcx [GenericArg<'tcx>],

    reachable_recursive_calls: Vec<Span>,
}

impl<'mir, 'tcx> Search<'mir, 'tcx> {
    /// Returns `true` if `func` refers to the function we are searching in.
    fn is_recursive_call(&self, func: &Operand<'tcx>, args: &[Operand<'tcx>]) -> bool {
        let Search { tcx, body, trait_substs, .. } = *self;
        // Resolving function type to a specific instance that is being called is expensive.  To
        // avoid the cost we check the number of arguments first, which is sufficient to reject
        // most of calls as non-recursive.
        if args.len() != body.arg_count {
            return false;
        }
        let caller = body.source.def_id();
        let param_env = tcx.param_env(caller);

        let func_ty = func.ty(body, tcx);
        if let ty::FnDef(callee, substs) = *func_ty.kind() {
            let normalized_substs = tcx.normalize_erasing_regions(param_env, substs);
            let (callee, call_substs) = if let Ok(Some(instance)) =
                Instance::resolve(tcx, param_env, callee, normalized_substs)
            {
                (instance.def_id(), instance.substs)
            } else {
                (callee, normalized_substs)
            };

            // FIXME(#57965): Make this work across function boundaries

            // If this is a trait fn, the substs on the trait have to match, or we might be
            // calling into an entirely different method (for example, a call from the default
            // method in the trait to `<A as Trait<B>>::method`, where `A` and/or `B` are
            // specific types).
            return callee == caller && &call_substs[..trait_substs.len()] == trait_substs;
        }

        false
    }
}

impl<'mir, 'tcx> TriColorVisitor<&'mir Body<'tcx>> for Search<'mir, 'tcx> {
    type BreakVal = NonRecursive;

    fn node_examined(
        &mut self,
        bb: BasicBlock,
        prior_status: Option<NodeStatus>,
    ) -> ControlFlow<Self::BreakVal> {
        // Back-edge in the CFG (loop).
        if let Some(NodeStatus::Visited) = prior_status {
            return ControlFlow::Break(NonRecursive);
        }

        match self.body[bb].terminator().kind {
            // These terminators return control flow to the caller.
            TerminatorKind::Abort
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::Resume
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Yield { .. } => ControlFlow::Break(NonRecursive),

            // A diverging InlineAsm is treated as non-recursing
            TerminatorKind::InlineAsm { destination, .. } => {
                if destination.is_some() {
                    ControlFlow::CONTINUE
                } else {
                    ControlFlow::Break(NonRecursive)
                }
            }

            // These do not.
            TerminatorKind::Assert { .. }
            | TerminatorKind::Call { .. }
            | TerminatorKind::Drop { .. }
            | TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::Goto { .. }
            | TerminatorKind::SwitchInt { .. } => ControlFlow::CONTINUE,
        }
    }

    fn node_settled(&mut self, bb: BasicBlock) -> ControlFlow<Self::BreakVal> {
        // When we examine a node for the last time, remember it if it is a recursive call.
        let terminator = self.body[bb].terminator();
        if let TerminatorKind::Call { func, args, .. } = &terminator.kind {
            if self.is_recursive_call(func, args) {
                self.reachable_recursive_calls.push(terminator.source_info.span);
            }
        }

        ControlFlow::CONTINUE
    }

    fn ignore_edge(&mut self, bb: BasicBlock, target: BasicBlock) -> bool {
        let terminator = self.body[bb].terminator();
        if terminator.unwind() == Some(&Some(target)) && terminator.successors().count() > 1 {
            return true;
        }
        // Don't traverse successors of recursive calls or false CFG edges.
        match self.body[bb].terminator().kind {
            TerminatorKind::Call { ref func, ref args, .. } => self.is_recursive_call(func, args),
            TerminatorKind::FalseEdge { imaginary_target, .. } => imaginary_target == target,
            _ => false,
        }
    }
}
