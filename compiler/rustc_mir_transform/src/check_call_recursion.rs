use std::ops::ControlFlow;

use rustc_data_structures::graph::iterate::{
    NodeStatus, TriColorDepthFirstSearch, TriColorVisitor,
};
use rustc_hir::LangItem;
use rustc_hir::def::DefKind;
use rustc_middle::mir::{self, BasicBlock, BasicBlocks, Body, Terminator, TerminatorKind};
use rustc_middle::ty::{self, GenericArg, GenericArgs, Instance, Ty, TyCtxt};
use rustc_session::lint::builtin::UNCONDITIONAL_RECURSION;
use rustc_span::Span;

use crate::errors::UnconditionalRecursion;
use crate::pass_manager::MirLint;

pub(super) struct CheckCallRecursion;

impl<'tcx> MirLint<'tcx> for CheckCallRecursion {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        let def_id = body.source.def_id().expect_local();

        if let DefKind::Fn | DefKind::AssocFn = tcx.def_kind(def_id) {
            // If this is trait/impl method, extract the trait's args.
            let trait_args = match tcx.trait_of_assoc(def_id.to_def_id()) {
                Some(trait_def_id) => {
                    let trait_args_count = tcx.generics_of(trait_def_id).count();
                    &GenericArgs::identity_for_item(tcx, def_id)[..trait_args_count]
                }
                _ => &[],
            };

            check_recursion(tcx, body, CallRecursion { trait_args })
        }
    }
}

/// Requires drop elaboration to have been performed.
pub(super) struct CheckDropRecursion;

impl<'tcx> MirLint<'tcx> for CheckDropRecursion {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        let def_id = body.source.def_id().expect_local();

        // First check if `body` is an `fn drop()` of `Drop`
        if let DefKind::AssocFn = tcx.def_kind(def_id)
        && let Some(impl_id) = tcx.trait_impl_of_assoc(def_id.to_def_id())
        && let trait_ref = tcx.impl_trait_ref(impl_id).unwrap()
        && tcx.is_lang_item(trait_ref.instantiate_identity().def_id, LangItem::Drop)
        // avoid erroneous `Drop` impls from causing ICEs below
        && let sig = tcx.fn_sig(def_id).instantiate_identity()
        && sig.inputs().skip_binder().len() == 1
        {
            // It was. Now figure out for what type `Drop` is implemented and then
            // check for recursion.
            if let ty::Ref(_, dropped_ty, _) =
                tcx.liberate_late_bound_regions(def_id.to_def_id(), sig.input(0)).kind()
            {
                check_recursion(tcx, body, RecursiveDrop { drop_for: *dropped_ty });
            }
        }
    }
}

fn check_recursion<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    classifier: impl TerminatorClassifier<'tcx>,
) {
    let def_id = body.source.def_id().expect_local();

    if let DefKind::Fn | DefKind::AssocFn = tcx.def_kind(def_id) {
        let mut vis = Search { tcx, body, classifier, reachable_recursive_calls: vec![] };
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
        let hir_id = tcx.local_def_id_to_hir_id(def_id);
        tcx.emit_node_span_lint(
            UNCONDITIONAL_RECURSION,
            hir_id,
            sp,
            UnconditionalRecursion { span: sp, call_sites: vis.reachable_recursive_calls },
        );
    }
}

trait TerminatorClassifier<'tcx> {
    fn is_recursive_terminator(
        &self,
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        terminator: &Terminator<'tcx>,
    ) -> bool;
}

struct NonRecursive;

struct Search<'mir, 'tcx, C: TerminatorClassifier<'tcx>> {
    tcx: TyCtxt<'tcx>,
    body: &'mir Body<'tcx>,
    classifier: C,

    reachable_recursive_calls: Vec<Span>,
}

struct CallRecursion<'tcx> {
    trait_args: &'tcx [GenericArg<'tcx>],
}

struct RecursiveDrop<'tcx> {
    /// The type that `Drop` is implemented for.
    drop_for: Ty<'tcx>,
}

impl<'tcx> TerminatorClassifier<'tcx> for CallRecursion<'tcx> {
    /// Returns `true` if `func` refers to the function we are searching in.
    fn is_recursive_terminator(
        &self,
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        terminator: &Terminator<'tcx>,
    ) -> bool {
        let TerminatorKind::Call { func, args, .. } = &terminator.kind else {
            return false;
        };

        // Resolving function type to a specific instance that is being called is expensive. To
        // avoid the cost we check the number of arguments first, which is sufficient to reject
        // most of calls as non-recursive.
        if args.len() != body.arg_count {
            return false;
        }
        let caller = body.source.def_id();
        let typing_env = body.typing_env(tcx);

        let func_ty = func.ty(body, tcx);
        if let ty::FnDef(callee, args) = *func_ty.kind() {
            let Ok(normalized_args) = tcx.try_normalize_erasing_regions(typing_env, args) else {
                return false;
            };
            let (callee, call_args) = if let Ok(Some(instance)) =
                Instance::try_resolve(tcx, typing_env, callee, normalized_args)
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
            return callee == caller && &call_args[..self.trait_args.len()] == self.trait_args;
        }

        false
    }
}

impl<'tcx> TerminatorClassifier<'tcx> for RecursiveDrop<'tcx> {
    fn is_recursive_terminator(
        &self,
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        terminator: &Terminator<'tcx>,
    ) -> bool {
        let TerminatorKind::Drop { place, .. } = &terminator.kind else { return false };

        let dropped_ty = place.ty(body, tcx).ty;
        dropped_ty == self.drop_for
    }
}

impl<'mir, 'tcx, C: TerminatorClassifier<'tcx>> TriColorVisitor<BasicBlocks<'tcx>>
    for Search<'mir, 'tcx, C>
{
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
            TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::UnwindResume
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Yield { .. } => ControlFlow::Break(NonRecursive),

            // A InlineAsm without targets (diverging and contains no labels)
            // is treated as non-recursing.
            TerminatorKind::InlineAsm { ref targets, .. } => {
                if !targets.is_empty() {
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

            // Note that tail call terminator technically returns to the caller,
            // but for purposes of this lint it makes sense to count it as possibly recursive,
            // since it's still a call.
            //
            // If this'll be repurposed for something else, this might need to be changed.
            TerminatorKind::TailCall { .. } => ControlFlow::Continue(()),
        }
    }

    fn node_settled(&mut self, bb: BasicBlock) -> ControlFlow<Self::BreakVal> {
        // When we examine a node for the last time, remember it if it is a recursive call.
        let terminator = self.body[bb].terminator();

        // FIXME(explicit_tail_calls): highlight tail calls as "recursive call site"
        //
        // We don't want to lint functions that recurse only through tail calls
        // (such as `fn g() { become () }`), so just adding `| TailCall { ... }`
        // here won't work.
        //
        // But at the same time we would like to highlight both calls in a function like
        // `fn f() { if false { become f() } else { f() } }`, so we need to figure something out.
        if self.classifier.is_recursive_terminator(self.tcx, self.body, terminator) {
            self.reachable_recursive_calls.push(terminator.source_info.span);
        }

        ControlFlow::Continue(())
    }

    fn ignore_edge(&mut self, bb: BasicBlock, target: BasicBlock) -> bool {
        let terminator = self.body[bb].terminator();
        let ignore_unwind = terminator.unwind() == Some(&mir::UnwindAction::Cleanup(target))
            && terminator.successors().count() > 1;
        if ignore_unwind || self.classifier.is_recursive_terminator(self.tcx, self.body, terminator)
        {
            return true;
        }
        match &terminator.kind {
            TerminatorKind::FalseEdge { imaginary_target, .. } => imaginary_target == &target,
            _ => false,
        }
    }
}
