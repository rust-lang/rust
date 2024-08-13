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

fn create_generic_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> ty::GenericArgsRef<'tcx> {
    let caller_generics = tcx.generics_of(def_id);
    let callee_generics = tcx.generics_of(sig_id);

    let caller_kind = fn_kind(tcx, def_id.into());
    let callee_kind = fn_kind(tcx, sig_id);
    // FIXME(fn_delegation): Support generics on associated delegation items.
    // Error will be reported in `check_constraints`.
    match (caller_kind, callee_kind) {
        (FnKind::Free, _) => {
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
            let mut remap_table = RemapTable::default();
            for caller_param in &caller_generics.own_params {
                let callee_index =
                    callee_generics.param_def_id_to_index(tcx, caller_param.def_id).unwrap();
                remap_table.insert(callee_index, caller_param.index);
            }
            let mut folder = ParamIndexRemapper { tcx, remap_table };
            ty::GenericArgs::identity_for_item(tcx, sig_id).fold_with(&mut folder)
        }
        // FIXME(fn_delegation): Only `Self` param supported here.
        (FnKind::AssocTraitImpl, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::AssocTrait) => {
            let parent = tcx.parent(def_id.into());
            let self_ty = tcx.type_of(parent).instantiate_identity();
            let generic_self_ty = ty::GenericArg::from(self_ty);
            tcx.mk_args_from_iter(std::iter::once(generic_self_ty))
        }
        _ => ty::GenericArgs::identity_for_item(tcx, sig_id),
    }
}

pub(crate) fn inherit_generics_for_delegation_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> Option<ty::Generics> {
    // FIXME(fn_delegation): Support generics on associated delegation items.
    // Error will be reported in `check_constraints`.
    if fn_kind(tcx, def_id.into()) != FnKind::Free {
        return None;
    }

    let mut own_params = vec![];

    let callee_generics = tcx.generics_of(sig_id);
    if let Some(parent_sig_id) = callee_generics.parent {
        let parent_sig_generics = tcx.generics_of(parent_sig_id);
        own_params.append(&mut parent_sig_generics.own_params.clone());
    }
    own_params.append(&mut callee_generics.own_params.clone());

    // Lifetimes go first.
    own_params.sort_by_key(|key| key.kind.is_ty_or_const());

    for (idx, param) in own_params.iter_mut().enumerate() {
        param.index = idx as u32;
        // Default parameters are not inherited: they are not allowed
        // in fn's.
        if let ty::GenericParamDefKind::Type { has_default, .. }
        | ty::GenericParamDefKind::Const { has_default, .. } = &mut param.kind
        {
            *has_default = false;
        }
    }

    let param_def_id_to_index =
        own_params.iter().map(|param| (param.def_id, param.index)).collect();

    Some(ty::Generics {
        parent: None,
        parent_count: 0,
        own_params,
        param_def_id_to_index,
        has_self: false,
        has_late_bound_regions: callee_generics.has_late_bound_regions,
        host_effect_index: callee_generics.host_effect_index,
    })
}

pub(crate) fn inherit_predicates_for_delegation_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> Option<ty::GenericPredicates<'tcx>> {
    // FIXME(fn_delegation): Support generics on associated delegation items.
    // Error will be reported in `check_constraints`.
    if fn_kind(tcx, def_id.into()) != FnKind::Free {
        return None;
    }

    let callee_predicates = tcx.predicates_of(sig_id);
    let args = create_generic_args(tcx, def_id, sig_id);

    let mut preds = vec![];
    if let Some(parent_id) = callee_predicates.parent {
        preds.extend(tcx.predicates_of(parent_id).instantiate_own(tcx, args));
    }
    preds.extend(callee_predicates.instantiate_own(tcx, args));

    Some(ty::GenericPredicates {
        parent: None,
        predicates: tcx.arena.alloc_from_iter(preds),
        effects_min_tys: ty::List::empty(),
    })
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
