use crate::{FnCtxt, TypeckRootCtxt};

use itertools::Itertools;
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::fold::{TypeFoldable, TypeFolder, TypeSuperFoldable};
use rustc_middle::ty::{
    self, GenericArgsRef, GenericParamDef, GenericParamDefKind, Ty, TyCtxt, TypeSuperVisitable,
    TypeVisitable, TypeVisitor,
};
use rustc_span::ErrorGuaranteed;
use rustc_trait_selection::infer::TyCtxtInferExt;

#[derive(Default)]
struct GenericDefsMap<'tcx> {
    // Generic arguments of each type encountered in delegation path.
    // For example in
    //
    // struct S<T> { ... }
    // reuse foo::<S<_>>;
    //
    // it will contain { DefId(foo): S<?0t>, DefId(S): ?0t }. Callee
    // signature and predicates will be substituted with this arguments
    // when the caller type is inherited.
    defs: FxIndexMap<DefId, GenericArgsRef<'tcx>>,
    // Maps inference variables to corresponding type parameter definitions.
    ty_defs: FxIndexMap<ty::TyVid, GenericParamDef>,
    // Maps inference variables to corresponding lifetime definitions.
    region_defs: FxIndexMap<ty::RegionVid, GenericParamDef>,
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

struct GenericDefsCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    info: GenericDefsMap<'tcx>,
    def_id: LocalDefId,
}

impl<'tcx> GenericDefsCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> GenericDefsCollector<'tcx> {
        GenericDefsCollector { tcx, info: GenericDefsMap::default(), def_id }
    }

    fn collect(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> GenericDefsMap<'tcx> {
        let mut collector = GenericDefsCollector::new(tcx, def_id);
        let caller_kind = fn_kind(tcx, def_id.into());
        // FIXME(fn_delegation): Support generics on associated delegation items.
        // Error was reported earlier in `check_constraints`.
        if caller_kind == FnKind::Free {
            let path = collector.check_call();
            path.visit_with(&mut collector);
        }

        collector.info
    }

    // Collect generic parameter definitions during callee type traversal according to
    // encountered inference variables.
    fn extract_info_from_def(&mut self, def_id: DefId, args: GenericArgsRef<'tcx>) {
        let generics = self.tcx.generics_of(def_id);

        for (idx, arg) in args.iter().enumerate() {
            if let Some(ty) = arg.as_type()
                && let ty::Infer(ty::InferTy::TyVar(ty_vid)) = ty.kind()
            {
                self.info.ty_defs.insert(*ty_vid, generics.param_at(idx, self.tcx).clone());
            } else if let Some(re) = arg.as_region()
                && let ty::RegionKind::ReVar(inf_re) = re.kind()
            {
                self.info.region_defs.insert(inf_re, generics.param_at(idx, self.tcx).clone());
            }
        }

        self.info.defs.insert(def_id, args);
    }

    // Extract callee type from the call path. Should only be used for
    // non-associated delegation items. For example in
    //
    // trait Trait<T> {
    //     fn foo<U>(&self, x: U, y: T);
    // }
    //
    // reuse <u16 as Trait<_>>::foo;
    //
    // it will return `FnDef(DefId(Trait::foo), [u16, ?0t, ?1t])`.
    fn check_call(&self) -> Ty<'tcx> {
        let body = self.tcx.hir().body_owned_by(self.def_id);
        let body = self.tcx.hir().body(body);

        let (expr, callee, args) = match body.value.kind {
            hir::ExprKind::Block(
                hir::Block {
                    expr: expr @ Some(hir::Expr { kind: hir::ExprKind::Call(callee, args), .. }),
                    ..
                },
                _,
            ) => (expr.unwrap(), callee, args),
            _ => unreachable!(),
        };

        let infcx = self.tcx.infer_ctxt().ignoring_regions().build();
        // FIXME: cycle error on `with_opaque_type_inference`
        let root_ctxt = TypeckRootCtxt::new_with_infcx(self.tcx, self.def_id, infcx);
        let param_env = ty::ParamEnv::empty();
        let fcx = FnCtxt::new(&root_ctxt, param_env, self.def_id);
        fcx.check_expr_with_expectation_and_args(
            callee,
            crate::expectation::Expectation::NoExpectation,
            args,
            Some(expr),
        )
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for GenericDefsCollector<'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        match ty.kind() {
            ty::Adt(adt, args) => self.extract_info_from_def(adt.did(), args),
            ty::FnDef(did, args) => self.extract_info_from_def(*did, args),
            _ => {}
        }

        ty.super_visit_with(self)
    }
}

struct DelegationArgFolder<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    info: &'a GenericDefsMap<'tcx>,
}

impl<'tcx, 'a> DelegationArgFolder<'tcx, 'a> {
    fn new(tcx: TyCtxt<'tcx>, info: &'a GenericDefsMap<'tcx>) -> DelegationArgFolder<'tcx, 'a> {
        DelegationArgFolder { tcx, info }
    }
}

impl<'tcx, 'a> TypeFolder<TyCtxt<'tcx>> for DelegationArgFolder<'tcx, 'a> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Infer(inf_ty) = ty.kind()
            && let ty::InferTy::TyVar(vid) = inf_ty
        {
            let param = self.info.ty_defs[vid].clone();
            let index = vid.as_u32() + self.info.region_defs.len() as u32;
            return Ty::new_param(self.tcx, index, param.name);
        };
        ty.super_fold_with(self)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        let ty::RegionKind::ReVar(rid) = &r.kind() else {
            return r;
        };
        let param = &self.info.region_defs[rid];
        ty::Region::new_early_param(
            self.tcx,
            ty::EarlyParamRegion { index: rid.as_u32(), name: param.name },
        )
    }
}

struct DelegationLowerer<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    info: &'a GenericDefsMap<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
}

impl<'tcx, 'a> DelegationLowerer<'tcx, 'a> {
    fn new(
        tcx: TyCtxt<'tcx>,
        def_id: LocalDefId,
        info: &'a GenericDefsMap<'tcx>,
    ) -> DelegationLowerer<'tcx, 'a> {
        DelegationLowerer { tcx, info, def_id, sig_id: tcx.hir().delegation_sig_id(def_id) }
    }

    fn fn_sig(&self) -> (&'tcx [Ty<'tcx>], Ty<'tcx>) {
        let caller_sig = self.tcx.fn_sig(self.sig_id);

        let caller_kind = fn_kind(self.tcx, self.def_id.into());
        let callee_kind = fn_kind(self.tcx, self.sig_id);

        // FIXME(fn_delegation): Support generics on associated delegation items.
        // Error was reported earlier in `check_constraints`.
        let sig = match (caller_kind, callee_kind) {
            (FnKind::Free, _) => {
                let mut args = self.info.defs[&self.sig_id];
                let mut folder = DelegationArgFolder::new(self.tcx, self.info);
                args = args.fold_with(&mut folder);

                caller_sig.instantiate(self.tcx, args)
            }
            // only `Self` param supported here
            (FnKind::AssocTraitImpl, FnKind::AssocTrait)
            | (FnKind::AssocInherentImpl, FnKind::AssocTrait) => {
                let parent = self.tcx.parent(self.def_id.into());
                let self_ty = self.tcx.type_of(parent).instantiate_identity();
                let generic_self_ty = ty::GenericArg::from(self_ty);
                let args = self.tcx.mk_args_from_iter(std::iter::once(generic_self_ty));
                caller_sig.instantiate(self.tcx, args)
            }
            _ => caller_sig.instantiate_identity(),
        };
        // Bound vars are also inherited from `sig_id`.
        // They will be rebound later in `lower_fn_ty`.
        let sig = sig.skip_binder();
        (sig.inputs(), sig.output())
    }

    // Type parameters may not be specified in callee path because they are inferred by compiler.
    // In contrast, constants must always be specified explicitly if the parameter is
    // not defined by default:
    //
    // fn foo<const N: i32>() {}
    //
    // reuse foo as bar;
    // // desugaring with inherited type info:
    // fn bar() {
    //   foo() // ERROR: cannot infer the value of the const parameter `N`
    // }
    //
    // Due to implementation limitations, the callee path is lowered without modifications.
    // As a result, we get a compilation error. Therefore, we do not inherit const parameters,
    // but they can be specified as generic arguments:
    //
    // fn foo<const N: i32>() {}
    //
    // reuse foo::<1> as bar;
    // // desugaring with inherited type info:
    // fn bar() {
    //   foo::<1>()
    // }
    fn generics_of(&self) -> Option<ty::Generics> {
        let callee_generics = self.tcx.generics_of(self.sig_id);

        let caller_kind = fn_kind(self.tcx, self.def_id.into());
        let callee_kind = fn_kind(self.tcx, self.sig_id);

        // FIXME(fn_delegation): Support generics on associated delegation items.
        // Error was reported earlier in `check_constraints`.
        match (caller_kind, callee_kind) {
            (FnKind::Free, _) => {
                let mut ty_params =
                    self.info.ty_defs.iter().map(|(_, param)| param.clone()).collect_vec();
                let mut region_params =
                    self.info.region_defs.iter().map(|(_, param)| param.clone()).collect_vec();

                region_params.append(&mut ty_params);
                let mut own_params = region_params;

                for (idx, param) in own_params.iter_mut().enumerate() {
                    param.index = idx as u32;
                    // Default type parameters are not inherited: they are not allowed
                    // in fn's.
                    if let GenericParamDefKind::Type { synthetic, .. } = param.kind {
                        param.kind = GenericParamDefKind::Type { has_default: false, synthetic }
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

    fn predicates_of(&self) -> Option<ty::GenericPredicates<'tcx>> {
        let caller_kind = fn_kind(self.tcx, self.def_id.into());
        let callee_kind = fn_kind(self.tcx, self.sig_id);

        // FIXME(fn_delegation): Support generics on associated delegation items.
        // Error was reported earlier in `check_constraints`.
        match (caller_kind, callee_kind) {
            (FnKind::Free, _) => {
                let mut predicates = vec![];
                for (def_id, args) in self.info.defs.iter() {
                    let def_predicates = self.tcx.explicit_predicates_of(def_id);
                    let mut instantiated_predicates =
                        def_predicates.instantiate(self.tcx, args).into_iter().collect_vec();
                    predicates.append(&mut instantiated_predicates);
                }

                let span = self.tcx.def_span(self.def_id);
                for predicate in predicates.iter_mut() {
                    predicate.1 = span;
                }

                let mut folder = DelegationArgFolder::new(self.tcx, self.info);
                let predicates = predicates.fold_with(&mut folder);
                let predicates = self.tcx.arena.alloc_from_iter(predicates);
                Some(ty::GenericPredicates { parent: None, predicates })
            }
            _ => None,
        }
    }
}

fn check_constraints<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> Result<(), ErrorGuaranteed> {
    let mut ret = Ok(());

    let span = tcx.def_span(def_id);
    let sig_id = tcx.hir().delegation_sig_id(def_id);

    let sig_span = tcx.def_span(sig_id);
    let mut emit = |descr| {
        ret = Err(tcx.dcx().emit_err(crate::errors::UnsupportedDelegation {
            span,
            descr,
            callee_span: sig_span,
        }));
    };

    if let Some(node) = tcx.hir().get_if_local(sig_id)
        && let Some(decl) = node.fn_decl()
        && let hir::FnRetTy::Return(ty) = decl.output
        && let hir::TyKind::InferDelegation(_, _) = ty.kind
    {
        emit("recursive delegation is not supported yet");
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

fn generate_error_results<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    err: ErrorGuaranteed,
) -> ty::LoweredDelegation<'tcx> {
    let sig_id = tcx.hir().delegation_sig_id(def_id);
    let sig_len = tcx.fn_arg_names(sig_id).len();
    let err_type = Ty::new_error(tcx, err);
    let inputs = tcx.arena.alloc_from_iter((0..sig_len).map(|_| err_type));
    ty::LoweredDelegation { inputs, output: err_type, generics: None, predicates: None }
}

pub fn lower_delegation_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> ty::LoweredDelegation<'tcx> {
    if let Err(err) = check_constraints(tcx, def_id) {
        return generate_error_results(tcx, def_id, err);
    }

    let info = GenericDefsCollector::collect(tcx, def_id);
    let delegation_resolver = DelegationLowerer::new(tcx, def_id, &info);
    let (inputs, output) = delegation_resolver.fn_sig();
    let generics = delegation_resolver.generics_of();
    let predicates = delegation_resolver.predicates_of();

    ty::LoweredDelegation { inputs, output, generics, predicates }
}
