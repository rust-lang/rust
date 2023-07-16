use crate::errors::UnconditionalRecursion;
use rustc_data_structures::graph::iterate::{
    NodeStatus, TriColorDepthFirstSearch, TriColorVisitor,
};
use rustc_hir::def::DefKind;
use rustc_middle::mir::{self, BasicBlock, BasicBlocks, Body, Operand, TerminatorKind};
use rustc_middle::ty::{self, Instance, TyCtxt};
use rustc_middle::ty::{GenericArg, GenericArgs};
use rustc_session::lint::builtin::UNCONDITIONAL_RECURSION;
use rustc_span::Span;
use std::ops::ControlFlow;

pub(crate) fn check<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
    let def_id = body.source.def_id().expect_local();

    if let DefKind::Fn | DefKind::AssocFn = tcx.def_kind(def_id) {
        // If this is trait/impl method, extract the trait's args.
        let trait_args = match tcx.trait_of_item(def_id.to_def_id()) {
            Some(trait_def_id) => {
                let trait_args_count = tcx.generics_of(trait_def_id).count();
                &GenericArgs::identity_for_item(tcx, def_id)[..trait_args_count]
            }
            _ => &[],
        };

        let mut vis = Search { tcx, body, reachable_recursive_calls: vec![], trait_args };
        if let Some(NonRecursive) =
            TriColorDepthFirstSearch::new(&body.basic_blocks).run_from_start(&mut vis)
        {
            return;
        }
        if vis.reachable_recursive_calls.is_empty() {
            return;
        }

        vis.reachable_recursive_calls.sort();

        let sp = tcx.def_span(def_id);
        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
        tcx.emit_spanned_lint(
            UNCONDITIONAL_RECURSION,
            hir_id,
            sp,
            UnconditionalRecursion { span: sp, call_sites: vis.reachable_recursive_calls },
        );
    }
}

struct NonRecursive;

struct Search<'mir, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'mir Body<'tcx>,
    trait_args: &'tcx [GenericArg<'tcx>],

    reachable_recursive_calls: Vec<Span>,
}

impl<'mir, 'tcx> Search<'mir, 'tcx> {
    /// Returns `true` if `func` refers to the function we are searching in.
    fn is_recursive_call(&self, func: &Operand<'tcx>, args: &[Operand<'tcx>]) -> bool {
        let Search { tcx, body, trait_args, .. } = *self;
        // Resolving function type to a specific instance that is being called is expensive. To
        // avoid the cost we check the number of arguments first, which is sufficient to reject
        // most of calls as non-recursive.
        if args.len() != body.arg_count {
            return false;
        }
        let caller = body.source.def_id();
        let param_env = tcx.param_env(caller);

        let func_ty = func.ty(body, tcx);
        if let ty::FnDef(callee, args) = *func_ty.kind() {
            let normalized_args = tcx.normalize_erasing_regions(param_env, args);
            let (callee, call_args) = if let Ok(Some(instance)) =
                Instance::resolve(tcx, param_env, callee, normalized_args)
            {
                (instance.def_id(), instance.args)
            } else {
                (callee, normalized_args)
            };

            // FIXME(#57965): Make this work across function boundaries

            // If this is a trait fn, the args on the trait have to match, or we might be
            // calling into an entirely different method (for example, a call from the default
            // method in the trait to `<A as Trait<B>>::method`, where `A` and/or `B` are
            // specific types).
            return callee == caller && &call_args[..trait_args.len()] == trait_args;
        }

        false
    }
}

impl<'mir, 'tcx> TriColorVisitor<BasicBlocks<'tcx>> for Search<'mir, 'tcx> {
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
            TerminatorKind::Terminate
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::Resume
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Yield { .. } => ControlFlow::Break(NonRecursive),

            // A diverging InlineAsm is treated as non-recursing
            TerminatorKind::InlineAsm { destination, .. } => {
                if destination.is_some() {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(NonRecursive)
                }
            }

            // These do not.
            TerminatorKind::Assert { .. }
            | TerminatorKind::Call { .. }
            | TerminatorKind::Drop { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::Goto { .. }
            | TerminatorKind::SwitchInt { .. } => ControlFlow::Continue(()),
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

        ControlFlow::Continue(())
    }

    fn ignore_edge(&mut self, bb: BasicBlock, target: BasicBlock) -> bool {
        let terminator = self.body[bb].terminator();
        if terminator.unwind() == Some(&mir::UnwindAction::Cleanup(target))
            && terminator.successors().count() > 1
        {
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
