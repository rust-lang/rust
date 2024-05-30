use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::fold::{TypeFoldable, TypeFolder, TypeSuperFoldable};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::ErrorGuaranteed;

type RemapTable = FxHashMap<u32, u32>;

struct IndicesFolder<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    remap_table: &'a RemapTable,
}

impl<'tcx, 'a> TypeFolder<TyCtxt<'tcx>> for IndicesFolder<'tcx, 'a> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Param(param) = ty.kind() {
            return Ty::new_param(self.tcx, self.remap_table[&param.index], param.name);
        }
        ty.super_fold_with(self)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        if let ty::ReEarlyParam(param) = r.kind() {
            return ty::Region::new_early_param(
                self.tcx,
                ty::EarlyParamRegion { index: self.remap_table[&param.index], name: param.name },
            );
        }
        r
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
    debug_assert!(matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn));

    let parent = tcx.parent(def_id);
    match tcx.def_kind(parent) {
        DefKind::Trait => FnKind::AssocTrait,
        DefKind::Impl { of_trait: true } => FnKind::AssocTraitImpl,
        DefKind::Impl { of_trait: false } => FnKind::AssocInherentImpl,
        _ => FnKind::Free,
    }
}

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
fn create_remap_table<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId, sig_id: DefId) -> RemapTable {
    let caller_generics = tcx.generics_of(def_id);
    let callee_generics = tcx.generics_of(sig_id);
    let mut remap_table: RemapTable = FxHashMap::default();
    for caller_param in &caller_generics.own_params {
        let callee_index = callee_generics.param_def_id_to_index(tcx, caller_param.def_id).unwrap();
        remap_table.insert(callee_index, caller_param.index);
    }
    remap_table
}

pub(crate) fn inherit_generics_for_delegation_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> Option<ty::Generics> {
    let caller_kind = fn_kind(tcx, def_id.into());
    let callee_kind = fn_kind(tcx, sig_id);

    // FIXME(fn_delegation): Support generics on associated delegation items.
    // Error was reported in `check_delegation_constraints`.
    match (caller_kind, callee_kind) {
        (FnKind::Free, _) => {
            let mut own_params = vec![];

            let callee_generics = tcx.generics_of(sig_id);
            if let Some(parent_sig_id) = callee_generics.parent {
                let parent_sig_generics = tcx.generics_of(parent_sig_id);
                own_params.append(&mut parent_sig_generics.own_params.clone());
            }
            own_params.append(&mut callee_generics.own_params.clone());

            // lifetimes go first
            own_params.sort_by_key(|key| key.kind.is_ty_or_const());

            for (idx, param) in own_params.iter_mut().enumerate() {
                param.index = idx as u32;
                // Default type parameters are not inherited: they are not allowed
                // in fn's.
                if let ty::GenericParamDefKind::Type { synthetic, .. } = param.kind {
                    param.kind = ty::GenericParamDefKind::Type { has_default: false, synthetic }
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
                host_effect_index: None,
            })
        }
        _ => None,
    }
}

pub(crate) fn inherit_predicates_for_delegation_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> Option<ty::GenericPredicates<'tcx>> {
    let caller_kind = fn_kind(tcx, def_id.into());
    let callee_kind = fn_kind(tcx, sig_id);

    // FIXME(fn_delegation): Support generics on associated delegation items.
    // Error was reported in `check_delegation_constraints`.
    match (caller_kind, callee_kind) {
        (FnKind::Free, _) => {
            let mut predicates = vec![];
            let callee_predicates = tcx.predicates_of(sig_id);
            if let Some(parent_sig_id) = callee_predicates.parent {
                let parent_sig_predicates = tcx.predicates_of(parent_sig_id);
                predicates.extend_from_slice(parent_sig_predicates.predicates);
            }
            predicates.extend_from_slice(callee_predicates.predicates);

            let remap_table = create_remap_table(tcx, def_id, sig_id);
            let mut folder = IndicesFolder { tcx, remap_table: &remap_table };
            let predicates = predicates.fold_with(&mut folder);

            Some(ty::GenericPredicates {
                parent: None,
                predicates: tcx.arena.alloc_from_iter(predicates),
            })
        }
        _ => None,
    }
}

fn check_constraints<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> Result<(), ErrorGuaranteed> {
    let mut ret = Ok(());
    let sig_id = tcx.hir().delegation_sig_id(def_id);

    let mut emit = |descr| {
        ret = Err(tcx.dcx().emit_err(crate::errors::UnsupportedDelegation {
            span: tcx.def_span(def_id),
            descr,
            callee_span: tcx.def_span(sig_id),
        }));
    };

    if let Some(local_sig_id) = sig_id.as_local()
        && tcx.hir().opt_delegation_sig_id(local_sig_id).is_some()
    {
        emit("recursive delegation");
    }

    let caller_kind = fn_kind(tcx, def_id.into());
    if caller_kind != FnKind::Free {
        let sig_generics = tcx.generics_of(sig_id);
        let parent = tcx.parent(def_id.into());
        let parent_generics = tcx.generics_of(parent);

        let parent_is_trait = (tcx.def_kind(parent) == DefKind::Trait) as usize;
        let sig_has_self = sig_generics.has_self as usize;

        if sig_generics.count() > sig_has_self || parent_generics.count() > parent_is_trait {
            emit("early bound generics are not supported for associated delegation items");
        }
    }

    ret
}

pub(crate) fn inherit_sig_for_delegation_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> &'tcx [Ty<'tcx>] {
    let sig_id = tcx.hir().delegation_sig_id(def_id);
    let caller_sig = tcx.fn_sig(sig_id);
    if let Err(err) = check_constraints(tcx, def_id) {
        let sig_len = caller_sig.instantiate_identity().skip_binder().inputs().len() + 1;
        let err_type = Ty::new_error(tcx, err);
        return tcx.arena.alloc_from_iter((0..sig_len).map(|_| err_type));
    }

    let caller_kind = fn_kind(tcx, def_id.into());
    let callee_kind = fn_kind(tcx, sig_id);

    // FIXME(fn_delegation): Support generics on associated delegation items.
    // Error was reported in `check_constraints`.
    let sig = match (caller_kind, callee_kind) {
        (FnKind::Free, _) => {
            let remap_table = create_remap_table(tcx, def_id, sig_id);
            let mut folder = IndicesFolder { tcx, remap_table: &remap_table };
            caller_sig.instantiate_identity().fold_with(&mut folder)
        }
        // only `Self` param supported here
        (FnKind::AssocTraitImpl, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::AssocTrait) => {
            let parent = tcx.parent(def_id.into());
            let self_ty = tcx.type_of(parent).instantiate_identity();
            let generic_self_ty = ty::GenericArg::from(self_ty);
            let args = tcx.mk_args_from_iter(std::iter::once(generic_self_ty));
            caller_sig.instantiate(tcx, args)
        }
        _ => caller_sig.instantiate_identity(),
    };
    // Bound vars are also inherited from `sig_id`.
    // They will be rebound later in `lower_fn_ty`.
    let sig = sig.skip_binder();
    let sig_it = sig.inputs().iter().cloned().chain(std::iter::once(sig.output()));
    tcx.arena.alloc_from_iter(sig_it)
}
