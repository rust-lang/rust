//! Common logic for borrowck use-after-move errors when moved into a `fn(self)`,
//! as well as errors when attempting to call a non-const function in a const
//! context.

use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItemGroup;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, AssocItemContainer, DefIdTree, Instance, ParamEnv, Ty, TyCtxt};
use rustc_span::symbol::Ident;
use rustc_span::{sym, DesugaringKind, Span};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CallDesugaringKind {
    /// for _ in x {} calls x.into_iter()
    ForLoopIntoIter,
    /// x? calls x.branch()
    QuestionBranch,
    /// x? calls type_of(x)::from_residual()
    QuestionFromResidual,
    /// try { ..; x } calls type_of(x)::from_output(x)
    TryBlockFromOutput,
}

impl CallDesugaringKind {
    pub fn trait_def_id(self, tcx: TyCtxt<'_>) -> DefId {
        match self {
            Self::ForLoopIntoIter => tcx.get_diagnostic_item(sym::IntoIterator).unwrap(),
            Self::QuestionBranch | Self::TryBlockFromOutput => {
                tcx.lang_items().try_trait().unwrap()
            }
            Self::QuestionFromResidual => tcx.get_diagnostic_item(sym::FromResidual).unwrap(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CallKind<'tcx> {
    /// A normal method call of the form `receiver.foo(a, b, c)`
    Normal {
        self_arg: Option<Ident>,
        desugaring: Option<(CallDesugaringKind, Ty<'tcx>)>,
        /// Whether the self type of the method call has an `.as_ref()` method.
        /// Used for better diagnostics.
        is_option_or_result: bool,
    },
    /// A call to `Fn(..)::call(..)`, desugared from `my_closure(a, b, c)`
    FnCall { fn_trait_id: DefId, self_ty: Ty<'tcx> },
    /// A call to an operator trait, desuraged from operator syntax (e.g. `a << b`)
    Operator { self_arg: Option<Ident>, trait_id: DefId, self_ty: Ty<'tcx> },
    DerefCoercion {
        /// The `Span` of the `Target` associated type
        /// in the `Deref` impl we are using.
        deref_target: Span,
        /// The type `T::Deref` we are dereferencing to
        deref_target_ty: Ty<'tcx>,
        self_ty: Ty<'tcx>,
    },
}

pub fn call_kind<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    method_did: DefId,
    method_substs: SubstsRef<'tcx>,
    fn_call_span: Span,
    from_hir_call: bool,
    self_arg: Option<Ident>,
) -> CallKind<'tcx> {
    let parent = tcx.opt_associated_item(method_did).and_then(|assoc| match assoc.container {
        AssocItemContainer::ImplContainer(impl_did) => tcx.trait_id_of_impl(impl_did),
        AssocItemContainer::TraitContainer(trait_did) => Some(trait_did),
    });

    let fn_call = parent
        .and_then(|p| tcx.lang_items().group(LangItemGroup::Fn).iter().find(|did| **did == p));

    let operator = (!from_hir_call)
        .then(|| parent)
        .flatten()
        .and_then(|p| tcx.lang_items().group(LangItemGroup::Op).iter().find(|did| **did == p));

    let is_deref = !from_hir_call && tcx.is_diagnostic_item(sym::deref_method, method_did);

    // Check for a 'special' use of 'self' -
    // an FnOnce call, an operator (e.g. `<<`), or a
    // deref coercion.
    let kind = if let Some(&trait_id) = fn_call {
        Some(CallKind::FnCall { fn_trait_id: trait_id, self_ty: method_substs.type_at(0) })
    } else if let Some(&trait_id) = operator {
        Some(CallKind::Operator { self_arg, trait_id, self_ty: method_substs.type_at(0) })
    } else if is_deref {
        let deref_target = tcx.get_diagnostic_item(sym::deref_target).and_then(|deref_target| {
            Instance::resolve(tcx, param_env, deref_target, method_substs).transpose()
        });
        if let Some(Ok(instance)) = deref_target {
            let deref_target_ty = instance.ty(tcx, param_env);
            Some(CallKind::DerefCoercion {
                deref_target: tcx.def_span(instance.def_id()),
                deref_target_ty,
                self_ty: method_substs.type_at(0),
            })
        } else {
            None
        }
    } else {
        None
    };

    kind.unwrap_or_else(|| {
        // This isn't a 'special' use of `self`
        debug!(?method_did, ?fn_call_span);
        let desugaring = if Some(method_did) == tcx.lang_items().into_iter_fn()
            && fn_call_span.desugaring_kind() == Some(DesugaringKind::ForLoop)
        {
            Some((CallDesugaringKind::ForLoopIntoIter, method_substs.type_at(0)))
        } else if fn_call_span.desugaring_kind() == Some(DesugaringKind::QuestionMark) {
            if Some(method_did) == tcx.lang_items().branch_fn() {
                Some((CallDesugaringKind::QuestionBranch, method_substs.type_at(0)))
            } else if Some(method_did) == tcx.lang_items().from_residual_fn() {
                Some((CallDesugaringKind::QuestionFromResidual, method_substs.type_at(0)))
            } else {
                None
            }
        } else if Some(method_did) == tcx.lang_items().from_output_fn()
            && fn_call_span.desugaring_kind() == Some(DesugaringKind::TryBlock)
        {
            Some((CallDesugaringKind::TryBlockFromOutput, method_substs.type_at(0)))
        } else {
            None
        };
        let parent_self_ty = tcx
            .parent(method_did)
            .filter(|did| tcx.def_kind(*did) == rustc_hir::def::DefKind::Impl)
            .and_then(|did| match tcx.type_of(did).kind() {
                ty::Adt(def, ..) => Some(def.did),
                _ => None,
            });
        let is_option_or_result = parent_self_ty.map_or(false, |def_id| {
            matches!(tcx.get_diagnostic_name(def_id), Some(sym::Option | sym::Result))
        });
        CallKind::Normal { self_arg, desugaring, is_option_or_result }
    })
}
