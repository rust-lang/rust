//! Support inheriting generic parameters and predicates for function delegation.
//!
//! For more information about delegation design, see the tracking issue #118212.

use std::assert_matches::debug_assert_matches;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
};
use rustc_span::{ErrorGuaranteed, Span};

type RemapTable = FxHashMap<u32, u32>;

struct ParamIndexRemapper<'tcx> {
    tcx: TyCtxt<'tcx>,
    remap_table: RemapTable,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for ParamIndexRemapper<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !ty.has_param() {
            return ty;
        }

        if let ty::Param(param) = ty.kind()
            && let Some(index) = self.remap_table.get(&param.index)
        {
            return Ty::new_param(self.tcx, *index, param.name);
        }
        ty.super_fold_with(self)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        if let ty::ReEarlyParam(param) = r.kind()
            && let Some(index) = self.remap_table.get(&param.index).copied()
        {
            return ty::Region::new_early_param(
                self.tcx,
                ty::EarlyParamRegion { index, name: param.name },
            );
        }
        r
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if let ty::ConstKind::Param(param) = ct.kind()
            && let Some(idx) = self.remap_table.get(&param.index)
        {
            let param = ty::ParamConst::new(*idx, param.name);
            return ty::Const::new_param(self.tcx, param);
        }
        ct.super_fold_with(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FnKind {
    Free,
    AssocInherentImpl,
    AssocTrait,
    AssocTraitImpl,
}

fn fn_kind<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> FnKind {
    debug_assert_matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn);

    let parent = tcx.parent(def_id);
    match tcx.def_kind(parent) {
        DefKind::Trait => FnKind::AssocTrait,
        DefKind::Impl { of_trait: true } => FnKind::AssocTraitImpl,
        DefKind::Impl { of_trait: false } => FnKind::AssocInherentImpl,
        _ => FnKind::Free,
    }
}

/// Given the current context(caller and callee `FnKind`), it specifies
/// the policy of predicates and generic parameters inheritance.
#[derive(Clone, Copy, Debug, PartialEq)]
enum InheritanceKind {
    /// Copying all predicates and parameters, including those of the parent
    /// container.
    ///
    /// Boolean value defines whether the `Self` parameter or `Self: Trait`
    /// predicate are copied. It's always equal to `false` except when
    /// delegating from a free function to a trait method.
    ///
    /// FIXME(fn_delegation): This often leads to type inference
    /// errors. Support providing generic arguments or restrict use sites.
    WithParent(bool),
    /// The trait implementation should be compatible with the original trait.
    /// Therefore, for trait implementations only the method's own parameters
    /// and predicates are copied.
    Own,
}

fn build_generics<'tcx>(
    tcx: TyCtxt<'tcx>,
    sig_id: DefId,
    parent: Option<DefId>,
    inh_kind: InheritanceKind,
) -> ty::Generics {
    let mut own_params = vec![];

    let sig_generics = tcx.generics_of(sig_id);
    if let InheritanceKind::WithParent(has_self) = inh_kind
        && let Some(parent_def_id) = sig_generics.parent
    {
        let sig_parent_generics = tcx.generics_of(parent_def_id);
        own_params.append(&mut sig_parent_generics.own_params.clone());
        if !has_self {
            own_params.remove(0);
        }
    }
    own_params.append(&mut sig_generics.own_params.clone());

    // Lifetime parameters must be declared before type and const parameters.
    // Therefore, When delegating from a free function to a associated function,
    // generic parameters need to be reordered:
    //
    // trait Trait<'a, A> {
    //     fn foo<'b, B>(...) {...}
    // }
    //
    // reuse Trait::foo;
    // desugaring:
    // fn foo<'a, 'b, This: Trait<'a, A>, A, B>(...) {
    //     Trait::foo(...)
    // }
    own_params.sort_by_key(|key| key.kind.is_ty_or_const());

    let param_def_id_to_index =
        own_params.iter().map(|param| (param.def_id, param.index)).collect();

    let (parent_count, has_self) = if let Some(def_id) = parent {
        let parent_generics = tcx.generics_of(def_id);
        let parent_kind = tcx.def_kind(def_id);
        (parent_generics.count(), parent_kind == DefKind::Trait)
    } else {
        (0, false)
    };

    for (idx, param) in own_params.iter_mut().enumerate() {
        param.index = (idx + parent_count) as u32;
        // FIXME(fn_delegation): Default parameters are not inherited, because they are
        // not permitted in functions. Therefore, there are 2 options here:
        //
        // - We can create non-default generic parameters.
        // - We can substitute default parameters into the signature.
        //
        // At the moment, first option has been selected as the most general.
        if let ty::GenericParamDefKind::Type { has_default, .. }
        | ty::GenericParamDefKind::Const { has_default, .. } = &mut param.kind
        {
            *has_default = false;
        }
    }

    ty::Generics {
        parent,
        parent_count,
        own_params,
        param_def_id_to_index,
        has_self,
        has_late_bound_regions: sig_generics.has_late_bound_regions,
    }
}

fn build_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    sig_id: DefId,
    parent: Option<DefId>,
    inh_kind: InheritanceKind,
    args: ty::GenericArgsRef<'tcx>,
) -> ty::GenericPredicates<'tcx> {
    struct PredicatesCollector<'tcx> {
        tcx: TyCtxt<'tcx>,
        preds: Vec<(ty::Clause<'tcx>, Span)>,
        args: ty::GenericArgsRef<'tcx>,
    }

    impl<'tcx> PredicatesCollector<'tcx> {
        fn new(tcx: TyCtxt<'tcx>, args: ty::GenericArgsRef<'tcx>) -> PredicatesCollector<'tcx> {
            PredicatesCollector { tcx, preds: vec![], args }
        }

        fn with_own_preds(
            mut self,
            f: impl Fn(DefId) -> ty::GenericPredicates<'tcx>,
            def_id: DefId,
        ) -> Self {
            let preds = f(def_id).instantiate_own(self.tcx, self.args);
            self.preds.extend(preds);
            self
        }

        fn with_preds(
            mut self,
            f: impl Fn(DefId) -> ty::GenericPredicates<'tcx> + Copy,
            def_id: DefId,
        ) -> Self {
            let preds = f(def_id);
            if let Some(parent_def_id) = preds.parent {
                self = self.with_own_preds(f, parent_def_id);
            }
            self.with_own_preds(f, def_id)
        }
    }
    let collector = PredicatesCollector::new(tcx, args);

    // `explicit_predicates_of` is used here to avoid copying `Self: Trait` predicate.
    // Note: `predicates_of` query can also add inferred outlives predicates, but that
    // is not the case here as `sig_id` is either a trait or a function.
    let preds = match inh_kind {
        InheritanceKind::WithParent(false) => {
            collector.with_preds(|def_id| tcx.explicit_predicates_of(def_id), sig_id)
        }
        InheritanceKind::WithParent(true) => {
            collector.with_preds(|def_id| tcx.predicates_of(def_id), sig_id)
        }
        InheritanceKind::Own => {
            collector.with_own_preds(|def_id| tcx.predicates_of(def_id), sig_id)
        }
    }
    .preds;

    ty::GenericPredicates { parent, predicates: tcx.arena.alloc_from_iter(preds) }
}

fn build_generic_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    sig_id: DefId,
    def_id: LocalDefId,
    args: ty::GenericArgsRef<'tcx>,
) -> ty::GenericArgsRef<'tcx> {
    let caller_generics = tcx.generics_of(def_id);
    let callee_generics = tcx.generics_of(sig_id);

    let mut remap_table = FxHashMap::default();
    for caller_param in &caller_generics.own_params {
        let callee_index = callee_generics.param_def_id_to_index(tcx, caller_param.def_id).unwrap();
        remap_table.insert(callee_index, caller_param.index);
    }

    let mut folder = ParamIndexRemapper { tcx, remap_table };
    args.fold_with(&mut folder)
}

fn create_generic_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> ty::GenericArgsRef<'tcx> {
    let caller_kind = fn_kind(tcx, def_id.into());
    let callee_kind = fn_kind(tcx, sig_id);
    match (caller_kind, callee_kind) {
        (FnKind::Free, FnKind::Free)
        | (FnKind::Free, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::Free)
        | (FnKind::AssocTrait, FnKind::Free)
        | (FnKind::AssocTrait, FnKind::AssocTrait) => {
            let args = ty::GenericArgs::identity_for_item(tcx, sig_id);
            build_generic_args(tcx, sig_id, def_id, args)
        }

        (FnKind::AssocTraitImpl, FnKind::AssocTrait) => {
            let callee_generics = tcx.generics_of(sig_id);
            let parent = tcx.parent(def_id.into());
            let parent_args =
                tcx.impl_trait_header(parent).unwrap().trait_ref.instantiate_identity().args;

            let trait_args = ty::GenericArgs::identity_for_item(tcx, sig_id);
            let method_args = tcx.mk_args(&trait_args[callee_generics.parent_count..]);
            let method_args = build_generic_args(tcx, sig_id, def_id, method_args);

            tcx.mk_args_from_iter(parent_args.iter().chain(method_args))
        }

        (FnKind::AssocInherentImpl, FnKind::AssocTrait) => {
            let parent = tcx.parent(def_id.into());
            let self_ty = tcx.type_of(parent).instantiate_identity();
            let generic_self_ty = ty::GenericArg::from(self_ty);

            let trait_args = ty::GenericArgs::identity_for_item(tcx, sig_id);
            let trait_args = build_generic_args(tcx, sig_id, def_id, trait_args);

            let args = std::iter::once(generic_self_ty).chain(trait_args.iter().skip(1));
            tcx.mk_args_from_iter(args)
        }

        // For trait impl's `sig_id` is always equal to the corresponding trait method.
        // For inherent methods delegation is not yet supported.
        (FnKind::AssocTraitImpl, _)
        | (_, FnKind::AssocTraitImpl)
        | (_, FnKind::AssocInherentImpl) => unreachable!(),
    }
}

// FIXME(fn_delegation): Move generics inheritance to the AST->HIR lowering.
// For now, generic parameters are not propagated to the generated call,
// which leads to inference errors:
//
// fn foo<T>(x: i32) {}
//
// reuse foo as bar;
// desugaring:
// fn bar<T>() {
//   foo::<_>() // ERROR: type annotations needed
// }
pub(crate) fn inherit_generics_for_delegation_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> ty::Generics {
    let caller_kind = fn_kind(tcx, def_id.into());
    let callee_kind = fn_kind(tcx, sig_id);
    match (caller_kind, callee_kind) {
        (FnKind::Free, FnKind::Free) | (FnKind::Free, FnKind::AssocTrait) => {
            build_generics(tcx, sig_id, None, InheritanceKind::WithParent(true))
        }

        (FnKind::AssocTraitImpl, FnKind::AssocTrait) => {
            build_generics(tcx, sig_id, Some(tcx.parent(def_id.into())), InheritanceKind::Own)
        }

        (FnKind::AssocInherentImpl, FnKind::AssocTrait)
        | (FnKind::AssocTrait, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::Free)
        | (FnKind::AssocTrait, FnKind::Free) => build_generics(
            tcx,
            sig_id,
            Some(tcx.parent(def_id.into())),
            InheritanceKind::WithParent(false),
        ),

        // For trait impl's `sig_id` is always equal to the corresponding trait method.
        // For inherent methods delegation is not yet supported.
        (FnKind::AssocTraitImpl, _)
        | (_, FnKind::AssocTraitImpl)
        | (_, FnKind::AssocInherentImpl) => unreachable!(),
    }
}

pub(crate) fn inherit_predicates_for_delegation_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> ty::GenericPredicates<'tcx> {
    let args = create_generic_args(tcx, def_id, sig_id);
    let caller_kind = fn_kind(tcx, def_id.into());
    let callee_kind = fn_kind(tcx, sig_id);
    match (caller_kind, callee_kind) {
        (FnKind::Free, FnKind::Free) | (FnKind::Free, FnKind::AssocTrait) => {
            build_predicates(tcx, sig_id, None, InheritanceKind::WithParent(true), args)
        }

        (FnKind::AssocTraitImpl, FnKind::AssocTrait) => build_predicates(
            tcx,
            sig_id,
            Some(tcx.parent(def_id.into())),
            InheritanceKind::Own,
            args,
        ),

        (FnKind::AssocInherentImpl, FnKind::AssocTrait)
        | (FnKind::AssocTrait, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::Free)
        | (FnKind::AssocTrait, FnKind::Free) => build_predicates(
            tcx,
            sig_id,
            Some(tcx.parent(def_id.into())),
            InheritanceKind::WithParent(false),
            args,
        ),

        // For trait impl's `sig_id` is always equal to the corresponding trait method.
        // For inherent methods delegation is not yet supported.
        (FnKind::AssocTraitImpl, _)
        | (_, FnKind::AssocTraitImpl)
        | (_, FnKind::AssocInherentImpl) => unreachable!(),
    }
}

fn check_constraints<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> Result<(), ErrorGuaranteed> {
    let mut ret = Ok(());

    let mut emit = |descr| {
        ret = Err(tcx.dcx().emit_err(crate::errors::UnsupportedDelegation {
            span: tcx.def_span(def_id),
            descr,
            callee_span: tcx.def_span(sig_id),
        }));
    };

    if let Some(local_sig_id) = sig_id.as_local()
        && tcx.hir_opt_delegation_sig_id(local_sig_id).is_some()
    {
        emit("recursive delegation is not supported yet");
    }

    if tcx.fn_sig(sig_id).skip_binder().skip_binder().c_variadic {
        // See issue #127443 for explanation.
        emit("delegation to C-variadic functions is not allowed");
    }

    ret
}

pub(crate) fn inherit_sig_for_delegation_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> &'tcx [Ty<'tcx>] {
    let sig_id = tcx.hir_opt_delegation_sig_id(def_id).unwrap();
    let caller_sig = tcx.fn_sig(sig_id);
    if let Err(err) = check_constraints(tcx, def_id, sig_id) {
        let sig_len = caller_sig.instantiate_identity().skip_binder().inputs().len() + 1;
        let err_type = Ty::new_error(tcx, err);
        return tcx.arena.alloc_from_iter((0..sig_len).map(|_| err_type));
    }
    let args = create_generic_args(tcx, def_id, sig_id);

    // Bound vars are also inherited from `sig_id`.
    // They will be rebound later in `lower_fn_ty`.
    let sig = caller_sig.instantiate(tcx, args).skip_binder();
    let sig_iter = sig.inputs().iter().cloned().chain(std::iter::once(sig.output()));
    tcx.arena.alloc_from_iter(sig_iter)
}
