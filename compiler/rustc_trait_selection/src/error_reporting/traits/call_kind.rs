//! Common logic for borrowck use-after-move errors when moved into a `fn(self)`,
//! as well as errors when attempting to call a non-const function in a const
//! context.

use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::{LangItem, lang_items};
use rustc_middle::ty::{AssocItemContainer, GenericArgsRef, Instance, Ty, TyCtxt, TypingEnv};
use rustc_span::{DUMMY_SP, DesugaringKind, Ident, Span, sym};
use tracing::debug;

use crate::traits::specialization_graph;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CallDesugaringKind {
    /// for _ in x {} calls x.into_iter()
    ForLoopIntoIter,
    /// for _ in x {} calls iter.next()
    ForLoopNext,
    /// x? calls x.branch()
    QuestionBranch,
    /// x? calls type_of(x)::from_residual()
    QuestionFromResidual,
    /// try { ..; x } calls type_of(x)::from_output(x)
    TryBlockFromOutput,
    /// `.await` calls `IntoFuture::into_future`
    Await,
}

impl CallDesugaringKind {
    pub fn trait_def_id(self, tcx: TyCtxt<'_>) -> DefId {
        match self {
            Self::ForLoopIntoIter => tcx.get_diagnostic_item(sym::IntoIterator).unwrap(),
            Self::ForLoopNext => tcx.require_lang_item(LangItem::Iterator, DUMMY_SP),
            Self::QuestionBranch | Self::TryBlockFromOutput => {
                tcx.require_lang_item(LangItem::Try, DUMMY_SP)
            }
            Self::QuestionFromResidual => tcx.get_diagnostic_item(sym::FromResidual).unwrap(),
            Self::Await => tcx.get_diagnostic_item(sym::IntoFuture).unwrap(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CallKind<'tcx> {
    /// A normal method call of the form `receiver.foo(a, b, c)`
    Normal {
        self_arg: Option<Ident>,
        desugaring: Option<(CallDesugaringKind, Ty<'tcx>)>,
        method_did: DefId,
        method_args: GenericArgsRef<'tcx>,
    },
    /// A call to `Fn(..)::call(..)`, desugared from `my_closure(a, b, c)`
    FnCall { fn_trait_id: DefId, self_ty: Ty<'tcx> },
    /// A call to an operator trait, desugared from operator syntax (e.g. `a << b`)
    Operator { self_arg: Option<Ident>, trait_id: DefId, self_ty: Ty<'tcx> },
    DerefCoercion {
        /// The `Span` of the `Target` associated type
        /// in the `Deref` impl we are using.
        deref_target_span: Option<Span>,
        /// The type `T::Deref` we are dereferencing to
        deref_target_ty: Ty<'tcx>,
        self_ty: Ty<'tcx>,
    },
}

pub fn call_kind<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: TypingEnv<'tcx>,
    method_did: DefId,
    method_args: GenericArgsRef<'tcx>,
    fn_call_span: Span,
    from_hir_call: bool,
    self_arg: Option<Ident>,
) -> CallKind<'tcx> {
    let parent = tcx.opt_associated_item(method_did).and_then(|assoc| {
        let container_id = assoc.container_id(tcx);
        match assoc.container {
            AssocItemContainer::Impl => tcx.trait_id_of_impl(container_id),
            AssocItemContainer::Trait => Some(container_id),
        }
    });

    let fn_call = parent.filter(|&p| tcx.fn_trait_kind_from_def_id(p).is_some());

    let operator = if !from_hir_call && let Some(p) = parent {
        lang_items::OPERATORS.iter().filter_map(|&l| tcx.lang_items().get(l)).find(|&id| id == p)
    } else {
        None
    };

    // Check for a 'special' use of 'self' -
    // an FnOnce call, an operator (e.g. `<<`), or a
    // deref coercion.
    if let Some(trait_id) = fn_call {
        return CallKind::FnCall { fn_trait_id: trait_id, self_ty: method_args.type_at(0) };
    } else if let Some(trait_id) = operator {
        return CallKind::Operator { self_arg, trait_id, self_ty: method_args.type_at(0) };
    } else if !from_hir_call && tcx.is_diagnostic_item(sym::deref_method, method_did) {
        let deref_target_def_id =
            tcx.get_diagnostic_item(sym::deref_target).expect("deref method but no deref target");
        let deref_target_ty = tcx.normalize_erasing_regions(
            typing_env,
            Ty::new_projection(tcx, deref_target_def_id, method_args),
        );
        let deref_target_span = if let Ok(Some(instance)) =
            Instance::try_resolve(tcx, typing_env, method_did, method_args)
            && let instance_parent_def_id = tcx.parent(instance.def_id())
            && matches!(tcx.def_kind(instance_parent_def_id), DefKind::Impl { .. })
            && let Ok(instance) =
                specialization_graph::assoc_def(tcx, instance_parent_def_id, deref_target_def_id)
            && instance.is_final()
        {
            Some(tcx.def_span(instance.item.def_id))
        } else {
            None
        };
        return CallKind::DerefCoercion {
            deref_target_ty,
            deref_target_span,
            self_ty: method_args.type_at(0),
        };
    }

    // This isn't a 'special' use of `self`
    debug!(?method_did, ?fn_call_span);
    let desugaring = if tcx.is_lang_item(method_did, LangItem::IntoIterIntoIter)
        && fn_call_span.desugaring_kind() == Some(DesugaringKind::ForLoop)
    {
        Some((CallDesugaringKind::ForLoopIntoIter, method_args.type_at(0)))
    } else if tcx.is_lang_item(method_did, LangItem::IteratorNext)
        && fn_call_span.desugaring_kind() == Some(DesugaringKind::ForLoop)
    {
        Some((CallDesugaringKind::ForLoopNext, method_args.type_at(0)))
    } else if fn_call_span.desugaring_kind() == Some(DesugaringKind::QuestionMark) {
        if tcx.is_lang_item(method_did, LangItem::TryTraitBranch) {
            Some((CallDesugaringKind::QuestionBranch, method_args.type_at(0)))
        } else if tcx.is_lang_item(method_did, LangItem::TryTraitFromResidual) {
            Some((CallDesugaringKind::QuestionFromResidual, method_args.type_at(0)))
        } else {
            None
        }
    } else if tcx.is_lang_item(method_did, LangItem::TryTraitFromOutput)
        && fn_call_span.desugaring_kind() == Some(DesugaringKind::TryBlock)
    {
        Some((CallDesugaringKind::TryBlockFromOutput, method_args.type_at(0)))
    } else if fn_call_span.is_desugaring(DesugaringKind::Await) {
        Some((CallDesugaringKind::Await, method_args.type_at(0)))
    } else {
        None
    };
    CallKind::Normal { self_arg, desugaring, method_did, method_args }
}
