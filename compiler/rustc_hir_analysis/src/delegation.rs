use std::assert_matches::debug_assert_matches;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::fold::{TypeFoldable, TypeFolder, TypeSuperFoldable};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::ErrorGuaranteed;
use rustc_type_ir::visit::TypeVisitableExt;

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

struct GenericsBuilder<'tcx> {
    tcx: TyCtxt<'tcx>,
    sig_id: DefId,
    parent: Option<DefId>,
}

impl<'tcx> GenericsBuilder<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, sig_id: DefId) -> GenericsBuilder<'tcx> {
        GenericsBuilder { tcx, sig_id, parent: None }
    }

    fn build(self) -> ty::Generics {
        let mut own_params = vec![];

        let sig_generics = self.tcx.generics_of(self.sig_id);
        if let Some(parent_def_id) = sig_generics.parent {
            let sig_parent_generics = self.tcx.generics_of(parent_def_id);
            own_params.append(&mut sig_parent_generics.own_params.clone());
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

        for (idx, param) in own_params.iter_mut().enumerate() {
            param.index = idx as u32;
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
            parent: self.parent,
            parent_count: 0,
            own_params,
            param_def_id_to_index,
            has_self: false,
            has_late_bound_regions: sig_generics.has_late_bound_regions,
            host_effect_index: sig_generics.host_effect_index,
        }
    }
}

struct PredicatesBuilder<'tcx> {
    tcx: TyCtxt<'tcx>,
    args: ty::GenericArgsRef<'tcx>,
    parent: Option<DefId>,
    sig_id: DefId,
}

impl<'tcx> PredicatesBuilder<'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        args: ty::GenericArgsRef<'tcx>,
        sig_id: DefId,
    ) -> PredicatesBuilder<'tcx> {
        PredicatesBuilder { tcx, args, parent: None, sig_id }
    }

    fn build(self) -> ty::GenericPredicates<'tcx> {
        let mut preds = vec![];

        let sig_predicates = self.tcx.predicates_of(self.sig_id);
        if let Some(parent) = sig_predicates.parent {
            let sig_parent_preds = self.tcx.predicates_of(parent);
            preds.extend(sig_parent_preds.instantiate_own(self.tcx, self.args));
        }
        preds.extend(sig_predicates.instantiate_own(self.tcx, self.args));

        ty::GenericPredicates {
            parent: self.parent,
            predicates: self.tcx.arena.alloc_from_iter(preds),
            // FIXME(fn_delegation): Support effects.
            effects_min_tys: ty::List::empty(),
        }
    }
}

struct GenericArgsBuilder<'tcx> {
    tcx: TyCtxt<'tcx>,
    remap_table: RemapTable,
    sig_id: DefId,
    def_id: LocalDefId,
}

impl<'tcx> GenericArgsBuilder<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, sig_id: DefId, def_id: LocalDefId) -> GenericArgsBuilder<'tcx> {
        GenericArgsBuilder { tcx, remap_table: FxHashMap::default(), sig_id, def_id }
    }

    fn build_from_args(mut self, args: ty::GenericArgsRef<'tcx>) -> ty::GenericArgsRef<'tcx> {
        let caller_generics = self.tcx.generics_of(self.def_id);
        let callee_generics = self.tcx.generics_of(self.sig_id);

        for caller_param in &caller_generics.own_params {
            let callee_index =
                callee_generics.param_def_id_to_index(self.tcx, caller_param.def_id).unwrap();
            self.remap_table.insert(callee_index, caller_param.index);
        }

        let mut folder = ParamIndexRemapper { tcx: self.tcx, remap_table: self.remap_table };
        args.fold_with(&mut folder)
    }
}

fn create_generic_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> ty::GenericArgsRef<'tcx> {
    let builder = GenericArgsBuilder::new(tcx, sig_id, def_id);

    let caller_kind = fn_kind(tcx, def_id.into());
    let callee_kind = fn_kind(tcx, sig_id);

    match (caller_kind, callee_kind) {
        (FnKind::Free, FnKind::Free)
        | (FnKind::Free, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::Free)
        | (FnKind::AssocTrait, FnKind::Free)
        | (FnKind::AssocTrait, FnKind::AssocTrait) => {
            let args = ty::GenericArgs::identity_for_item(tcx, sig_id);
            builder.build_from_args(args)
        }
        // FIXME(fn_delegation): Only `Self` param supported here.
        (FnKind::AssocTraitImpl, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::AssocTrait) => {
            let parent = tcx.parent(def_id.into());
            let self_ty = tcx.type_of(parent).instantiate_identity();
            let generic_self_ty = ty::GenericArg::from(self_ty);
            tcx.mk_args_from_iter(std::iter::once(generic_self_ty))
        }
        // For trait impl's `sig_id` is always equal to the corresponding trait method.
        (FnKind::AssocTraitImpl, _)
        | (_, FnKind::AssocTraitImpl)
        // Delegation to inherent methods is not yet supported.
        | (_, FnKind::AssocInherentImpl) => unreachable!(),
    }
}

pub(crate) fn inherit_generics_for_delegation_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> Option<ty::Generics> {
    let builder = GenericsBuilder::new(tcx, sig_id);

    let caller_kind = fn_kind(tcx, def_id.into());
    let callee_kind = fn_kind(tcx, sig_id);

    // FIXME(fn_delegation): Support generics on associated delegation items.
    // Error will be reported in `check_constraints`.
    match (caller_kind, callee_kind) {
        (FnKind::Free, FnKind::Free)
        | (FnKind::Free, FnKind::AssocTrait) => Some(builder.build()),

        (FnKind::AssocTraitImpl, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::AssocTrait)
        | (FnKind::AssocTrait, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::Free)
        | (FnKind::AssocTrait, FnKind::Free) => None,

        // For trait impl's `sig_id` is always equal to the corresponding trait method.
        (FnKind::AssocTraitImpl, _)
        | (_, FnKind::AssocTraitImpl)
        // Delegation to inherent methods is not yet supported.
        | (_, FnKind::AssocInherentImpl) => unreachable!(),
    }
}

pub(crate) fn inherit_predicates_for_delegation_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> Option<ty::GenericPredicates<'tcx>> {
    let args = create_generic_args(tcx, def_id, sig_id);
    let builder = PredicatesBuilder::new(tcx, args, sig_id);

    let caller_kind = fn_kind(tcx, def_id.into());
    let callee_kind = fn_kind(tcx, sig_id);

    // FIXME(fn_delegation): Support generics on associated delegation items.
    // Error will be reported in `check_constraints`.
    match (caller_kind, callee_kind) {
        (FnKind::Free, FnKind::Free)
        | (FnKind::Free, FnKind::AssocTrait) => {
            Some(builder.build())
        }

        (FnKind::AssocTraitImpl, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::AssocTrait)
        | (FnKind::AssocTrait, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::Free)
        | (FnKind::AssocTrait, FnKind::Free) => None,

        // For trait impl's `sig_id` is always equal to the corresponding trait method.
        (FnKind::AssocTraitImpl, _)
        | (_, FnKind::AssocTraitImpl)
        // Delegation to inherent methods is not yet supported.
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

    if tcx.has_host_param(sig_id) {
        emit("delegation to a function with effect parameter is not supported yet");
    }

    if let Some(local_sig_id) = sig_id.as_local()
        && tcx.hir().opt_delegation_sig_id(local_sig_id).is_some()
    {
        emit("recursive delegation is not supported yet");
    }

    if fn_kind(tcx, def_id.into()) != FnKind::Free {
        let sig_generics = tcx.generics_of(sig_id);
        let parent = tcx.parent(def_id.into());
        let parent_generics = tcx.generics_of(parent);

        let parent_has_self = parent_generics.has_self as usize;
        let sig_has_self = sig_generics.has_self as usize;

        if sig_generics.count() > sig_has_self || parent_generics.count() > parent_has_self {
            emit("early bound generics are not supported for associated delegation items");
        }
    }

    ret
}

pub(crate) fn inherit_sig_for_delegation_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> &'tcx [Ty<'tcx>] {
    let sig_id = tcx.hir().opt_delegation_sig_id(def_id).unwrap();
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
