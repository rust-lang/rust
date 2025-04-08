use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::OpaqueTyOrigin;
use rustc_hir::def_id::LocalDefId;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_middle::ty::{
    self, DefiningScopeKind, GenericArgKind, GenericArgs, OpaqueTypeKey, TyCtxt, TypeVisitableExt,
    TypingMode, fold_regions,
};
use rustc_span::{ErrorGuaranteed, Span};

use crate::errors::NonGenericOpaqueTypeParam;
use crate::regions::OutlivesEnvironmentBuildExt;
use crate::traits::ObligationCtxt;

/// Opaque type parameter validity check as documented in the [rustc-dev-guide chapter].
///
/// [rustc-dev-guide chapter]:
/// https://rustc-dev-guide.rust-lang.org/opaque-types-region-infer-restrictions.html
pub fn check_opaque_type_parameter_valid<'tcx>(
    infcx: &InferCtxt<'tcx>,
    opaque_type_key: OpaqueTypeKey<'tcx>,
    span: Span,
    defining_scope_kind: DefiningScopeKind,
) -> Result<(), ErrorGuaranteed> {
    let tcx = infcx.tcx;
    let opaque_generics = tcx.generics_of(opaque_type_key.def_id);
    let opaque_env = LazyOpaqueTyEnv::new(tcx, opaque_type_key.def_id);
    let mut seen_params: FxIndexMap<_, Vec<_>> = FxIndexMap::default();

    // Avoid duplicate errors in case the opaque has already been malformed in
    // HIR typeck.
    if let DefiningScopeKind::MirBorrowck = defining_scope_kind {
        if let Err(guar) = infcx
            .tcx
            .type_of_opaque_hir_typeck(opaque_type_key.def_id)
            .instantiate_identity()
            .error_reported()
        {
            return Err(guar);
        }
    }

    for (i, arg) in opaque_type_key.iter_captured_args(tcx) {
        let arg_is_param = match arg.unpack() {
            GenericArgKind::Lifetime(lt) => match defining_scope_kind {
                DefiningScopeKind::HirTypeck => continue,
                DefiningScopeKind::MirBorrowck => {
                    matches!(lt.kind(), ty::ReEarlyParam(_) | ty::ReLateParam(_))
                        || (lt.is_static() && opaque_env.param_equal_static(i))
                }
            },
            GenericArgKind::Type(ty) => matches!(ty.kind(), ty::Param(_)),
            GenericArgKind::Const(ct) => matches!(ct.kind(), ty::ConstKind::Param(_)),
        };

        if arg_is_param {
            // Register if the same lifetime appears multiple times in the generic args.
            // There is an exception when the opaque type *requires* the lifetimes to be equal.
            // See [rustc-dev-guide chapter] § "An exception to uniqueness rule".
            let seen_where = seen_params.entry(arg).or_default();
            if !seen_where.first().is_some_and(|&prev_i| opaque_env.params_equal(i, prev_i)) {
                seen_where.push(i);
            }
        } else {
            // Prevent `fn foo() -> Foo<u32>` from being defining.
            let opaque_param = opaque_generics.param_at(i, tcx);
            let kind = opaque_param.kind.descr();

            opaque_env.param_is_error(i)?;

            return Err(infcx.dcx().emit_err(NonGenericOpaqueTypeParam {
                ty: arg,
                kind,
                span,
                param_span: tcx.def_span(opaque_param.def_id),
            }));
        }
    }

    for (_, indices) in seen_params {
        if indices.len() > 1 {
            let descr = opaque_generics.param_at(indices[0], tcx).kind.descr();
            let spans: Vec<_> = indices
                .into_iter()
                .map(|i| tcx.def_span(opaque_generics.param_at(i, tcx).def_id))
                .collect();
            return Err(infcx
                .dcx()
                .struct_span_err(span, "non-defining opaque type use in defining scope")
                .with_span_note(spans, format!("{descr} used multiple times"))
                .emit());
        }
    }

    Ok(())
}

/// Computes if an opaque type requires a lifetime parameter to be equal to
/// another one or to the `'static` lifetime.
/// These requirements are derived from the explicit and implied bounds.
struct LazyOpaqueTyEnv<'tcx> {
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,

    /// Equal parameters will have the same name. Computed Lazily.
    /// Example:
    ///     `type Opaque<'a: 'static, 'b: 'c, 'c: 'b> = impl Sized;`
    ///     Identity args: `['a, 'b, 'c]`
    ///     Canonical args: `['static, 'b, 'b]`
    canonical_args: std::cell::OnceCell<ty::GenericArgsRef<'tcx>>,
}

impl<'tcx> LazyOpaqueTyEnv<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> Self {
        Self { tcx, def_id, canonical_args: std::cell::OnceCell::new() }
    }

    fn param_equal_static(&self, param_index: usize) -> bool {
        self.get_canonical_args()[param_index].expect_region().is_static()
    }

    fn params_equal(&self, param1: usize, param2: usize) -> bool {
        let canonical_args = self.get_canonical_args();
        canonical_args[param1] == canonical_args[param2]
    }

    fn param_is_error(&self, param_index: usize) -> Result<(), ErrorGuaranteed> {
        self.get_canonical_args()[param_index].error_reported()
    }

    fn get_canonical_args(&self) -> ty::GenericArgsRef<'tcx> {
        if let Some(&canonical_args) = self.canonical_args.get() {
            return canonical_args;
        }

        let &Self { tcx, def_id, .. } = self;
        let origin = tcx.local_opaque_ty_origin(def_id);
        let parent = match origin {
            OpaqueTyOrigin::FnReturn { parent, .. }
            | OpaqueTyOrigin::AsyncFn { parent, .. }
            | OpaqueTyOrigin::TyAlias { parent, .. } => parent,
        };
        let param_env = tcx.param_env(parent);
        let args = GenericArgs::identity_for_item(tcx, parent).extend_to(
            tcx,
            def_id.to_def_id(),
            |param, _| {
                tcx.map_opaque_lifetime_to_parent_lifetime(param.def_id.expect_local()).into()
            },
        );

        // FIXME(#132279): It feels wrong to use `non_body_analysis` here given that we're
        // in a body here.
        let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
        let ocx = ObligationCtxt::new(&infcx);

        let wf_tys = ocx.assumed_wf_types(param_env, parent).unwrap_or_else(|_| {
            tcx.dcx().span_delayed_bug(tcx.def_span(def_id), "error getting implied bounds");
            Default::default()
        });
        let outlives_env = OutlivesEnvironment::new(&infcx, parent, param_env, wf_tys);

        let mut seen = vec![tcx.lifetimes.re_static];
        let canonical_args = fold_regions(tcx, args, |r1, _| {
            if r1.is_error() {
                r1
            } else if let Some(&r2) = seen.iter().find(|&&r2| {
                let free_regions = outlives_env.free_region_map();
                free_regions.sub_free_regions(tcx, r1, r2)
                    && free_regions.sub_free_regions(tcx, r2, r1)
            }) {
                r2
            } else {
                seen.push(r1);
                r1
            }
        });
        self.canonical_args.set(canonical_args).unwrap();
        canonical_args
    }
}
