use std::cell::RefCell;

use crate::coercion::CoerceMany;
use crate::gather_locals::GatherLocalsVisitor;
use crate::CoroutineTypes;
use crate::FnCtxt;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::intravisit::Visitor;
use rustc_hir::lang_items::LangItem;
use rustc_hir_analysis::check::{check_function_signature, fn_maybe_err};
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::RegionVariableOrigin;
use rustc_middle::ty::{self, Binder, Ty, TyCtxt};
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::sym;
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits;
use rustc_trait_selection::traits::{ObligationCause, ObligationCauseCode};

/// Helper used for fns and closures. Does the grungy work of checking a function
/// body and returns the function context used for that purpose, since in the case of a fn item
/// there is still a bit more to do.
///
/// * ...
/// * inherited: other fields inherited from the enclosing fn (if any)
#[instrument(skip(fcx, body), level = "debug")]
pub(super) fn check_fn<'a, 'tcx>(
    fcx: &mut FnCtxt<'a, 'tcx>,
    fn_sig: ty::FnSig<'tcx>,
    decl: &'tcx hir::FnDecl<'tcx>,
    fn_def_id: LocalDefId,
    body: &'tcx hir::Body<'tcx>,
    can_be_coroutine: Option<hir::Movability>,
    params_can_be_unsized: bool,
) -> Option<CoroutineTypes<'tcx>> {
    let fn_id = fcx.tcx.hir().local_def_id_to_hir_id(fn_def_id);

    let tcx = fcx.tcx;
    let hir = tcx.hir();

    let declared_ret_ty = fn_sig.output();

    let ret_ty =
        fcx.register_infer_ok_obligations(fcx.infcx.replace_opaque_types_with_inference_vars(
            declared_ret_ty,
            fn_def_id,
            decl.output.span(),
            fcx.param_env,
        ));

    fcx.ret_coercion = Some(RefCell::new(CoerceMany::new(ret_ty)));

    let span = body.value.span;

    fn_maybe_err(tcx, span, fn_sig.abi);

    if let Some(kind) = body.coroutine_kind
        && can_be_coroutine.is_some()
    {
        let yield_ty = match kind {
            hir::CoroutineKind::Gen(..) | hir::CoroutineKind::Coroutine => {
                let yield_ty = fcx.next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::TypeInference,
                    span,
                });
                fcx.require_type_is_sized(yield_ty, span, traits::SizedYieldType);
                yield_ty
            }
            hir::CoroutineKind::Async(..) => Ty::new_unit(tcx),
        };

        // Resume type defaults to `()` if the coroutine has no argument.
        let resume_ty = fn_sig.inputs().get(0).copied().unwrap_or_else(|| Ty::new_unit(tcx));

        fcx.resume_yield_tys = Some((resume_ty, yield_ty));
    }

    GatherLocalsVisitor::new(&fcx).visit_body(body);

    // C-variadic fns also have a `VaList` input that's not listed in `fn_sig`
    // (as it's created inside the body itself, not passed in from outside).
    let maybe_va_list = fn_sig.c_variadic.then(|| {
        let span = body.params.last().unwrap().span;
        let va_list_did = tcx.require_lang_item(LangItem::VaList, Some(span));
        let region = fcx.next_region_var(RegionVariableOrigin::MiscVariable(span));

        tcx.type_of(va_list_did).instantiate(tcx, &[region.into()])
    });

    // Add formal parameters.
    let inputs_hir = hir.fn_decl_by_hir_id(fn_id).map(|decl| &decl.inputs);
    let inputs_fn = fn_sig.inputs().iter().copied();
    for (idx, (param_ty, param)) in inputs_fn.chain(maybe_va_list).zip(body.params).enumerate() {
        // Check the pattern.
        let ty_span = try { inputs_hir?.get(idx)?.span };
        fcx.check_pat_top(&param.pat, param_ty, ty_span, None, None);

        // Check that argument is Sized.
        if !params_can_be_unsized {
            fcx.require_type_is_sized(
                param_ty,
                param.pat.span,
                // ty_span == binding_span iff this is a closure parameter with no type ascription,
                // or if it's an implicit `self` parameter
                traits::SizedArgumentType(
                    if ty_span == Some(param.span) && tcx.is_closure(fn_def_id.into()) {
                        None
                    } else {
                        ty_span
                    },
                ),
            );
        }

        fcx.write_ty(param.hir_id, param_ty);
    }

    fcx.typeck_results.borrow_mut().liberated_fn_sigs_mut().insert(fn_id, fn_sig);

    let return_or_body_span = match decl.output {
        hir::FnRetTy::DefaultReturn(_) => body.value.span,
        hir::FnRetTy::Return(ty) => ty.span,
    };
    fcx.require_type_is_sized(declared_ret_ty, return_or_body_span, traits::SizedReturnType);
    fcx.check_return_expr(&body.value, false);

    // We insert the deferred_coroutine_interiors entry after visiting the body.
    // This ensures that all nested coroutines appear before the entry of this coroutine.
    // resolve_coroutine_interiors relies on this property.
    let coroutine_ty = if let (Some(_), Some(coroutine_kind)) =
        (can_be_coroutine, body.coroutine_kind)
    {
        let interior = fcx
            .next_ty_var(TypeVariableOrigin { kind: TypeVariableOriginKind::MiscVariable, span });
        fcx.deferred_coroutine_interiors.borrow_mut().push((
            fn_def_id,
            body.id(),
            interior,
            coroutine_kind,
        ));

        let (resume_ty, yield_ty) = fcx.resume_yield_tys.unwrap();
        Some(CoroutineTypes {
            resume_ty,
            yield_ty,
            interior,
            movability: can_be_coroutine.unwrap(),
        })
    } else {
        None
    };

    // Finalize the return check by taking the LUB of the return types
    // we saw and assigning it to the expected return type. This isn't
    // really expected to fail, since the coercions would have failed
    // earlier when trying to find a LUB.
    let coercion = fcx.ret_coercion.take().unwrap().into_inner();
    let mut actual_return_ty = coercion.complete(&fcx);
    debug!("actual_return_ty = {:?}", actual_return_ty);
    if let ty::Dynamic(..) = declared_ret_ty.kind() {
        // We have special-cased the case where the function is declared
        // `-> dyn Foo` and we don't actually relate it to the
        // `fcx.ret_coercion`, so just substitute a type variable.
        actual_return_ty =
            fcx.next_ty_var(TypeVariableOrigin { kind: TypeVariableOriginKind::DynReturnFn, span });
        debug!("actual_return_ty replaced with {:?}", actual_return_ty);
    }

    // HACK(oli-obk, compiler-errors): We should be comparing this against
    // `declared_ret_ty`, but then anything uninferred would be inferred to
    // the opaque type itself. That again would cause writeback to assume
    // we have a recursive call site and do the sadly stabilized fallback to `()`.
    fcx.demand_suptype(span, ret_ty, actual_return_ty);

    // Check that a function marked as `#[panic_handler]` has signature `fn(&PanicInfo) -> !`
    if let Some(panic_impl_did) = tcx.lang_items().panic_impl()
        && panic_impl_did == fn_def_id.to_def_id()
    {
        check_panic_info_fn(tcx, panic_impl_did.expect_local(), fn_sig);
    }

    if let Some(lang_start_defid) = tcx.lang_items().start_fn()
        && lang_start_defid == fn_def_id.to_def_id()
    {
        check_lang_start_fn(tcx, fn_sig, fn_def_id);
    }

    coroutine_ty
}

fn check_panic_info_fn(tcx: TyCtxt<'_>, fn_id: LocalDefId, fn_sig: ty::FnSig<'_>) {
    let DefKind::Fn = tcx.def_kind(fn_id) else {
        let span = tcx.def_span(fn_id);
        tcx.sess.span_err(span, "should be a function");
        return;
    };

    let generic_counts = tcx.generics_of(fn_id).own_counts();
    if generic_counts.types != 0 {
        let span = tcx.def_span(fn_id);
        tcx.sess.span_err(span, "should have no type parameters");
    }
    if generic_counts.consts != 0 {
        let span = tcx.def_span(fn_id);
        tcx.sess.span_err(span, "should have no const parameters");
    }

    let Some(panic_info_did) = tcx.lang_items().panic_info() else {
        tcx.sess.err("language item required, but not found: `panic_info`");
        return;
    };

    // build type `for<'a, 'b> fn(&'a PanicInfo<'b>) -> !`
    let panic_info_ty = tcx.type_of(panic_info_did).instantiate(
        tcx,
        &[ty::GenericArg::from(ty::Region::new_bound(
            tcx,
            ty::INNERMOST,
            ty::BoundRegion { var: ty::BoundVar::from_u32(1), kind: ty::BrAnon },
        ))],
    );
    let panic_info_ref_ty = Ty::new_imm_ref(
        tcx,
        ty::Region::new_bound(
            tcx,
            ty::INNERMOST,
            ty::BoundRegion { var: ty::BoundVar::from_u32(0), kind: ty::BrAnon },
        ),
        panic_info_ty,
    );

    let bounds = tcx.mk_bound_variable_kinds(&[
        ty::BoundVariableKind::Region(ty::BrAnon),
        ty::BoundVariableKind::Region(ty::BrAnon),
    ]);
    let expected_sig = ty::Binder::bind_with_vars(
        tcx.mk_fn_sig([panic_info_ref_ty], tcx.types.never, false, fn_sig.unsafety, Abi::Rust),
        bounds,
    );

    check_function_signature(
        tcx,
        ObligationCause::new(
            tcx.def_span(fn_id),
            fn_id,
            ObligationCauseCode::LangFunctionType(sym::panic_impl),
        ),
        fn_id.into(),
        expected_sig,
    );
}

fn check_lang_start_fn<'tcx>(tcx: TyCtxt<'tcx>, fn_sig: ty::FnSig<'tcx>, def_id: LocalDefId) {
    // build type `fn(main: fn() -> T, argc: isize, argv: *const *const u8, sigpipe: u8)`

    // make a Ty for the generic on the fn for diagnostics
    // FIXME: make the lang item generic checks check for the right generic *kind*
    // for example `start`'s generic should be a type parameter
    let generics = tcx.generics_of(def_id);
    let fn_generic = generics.param_at(0, tcx);
    let generic_ty = Ty::new_param(tcx, fn_generic.index, fn_generic.name);
    let main_fn_ty = Ty::new_fn_ptr(
        tcx,
        Binder::dummy(tcx.mk_fn_sig([], generic_ty, false, hir::Unsafety::Normal, Abi::Rust)),
    );

    let expected_sig = ty::Binder::dummy(tcx.mk_fn_sig(
        [
            main_fn_ty,
            tcx.types.isize,
            Ty::new_imm_ptr(tcx, Ty::new_imm_ptr(tcx, tcx.types.u8)),
            tcx.types.u8,
        ],
        tcx.types.isize,
        false,
        fn_sig.unsafety,
        Abi::Rust,
    ));

    check_function_signature(
        tcx,
        ObligationCause::new(
            tcx.def_span(def_id),
            def_id,
            ObligationCauseCode::LangFunctionType(sym::start),
        ),
        def_id.into(),
        expected_sig,
    );
}
