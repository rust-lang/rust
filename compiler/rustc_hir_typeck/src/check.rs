use crate::coercion::CoerceMany;
use crate::errors::{
    LangStartIncorrectNumberArgs, LangStartIncorrectParam, LangStartIncorrectRetTy,
};
use crate::gather_locals::GatherLocalsVisitor;
use crate::FnCtxt;
use crate::GeneratorTypes;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::intravisit::Visitor;
use rustc_hir::lang_items::LangItem;
use rustc_hir_analysis::check::fn_maybe_err;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::RegionVariableOrigin;
use rustc_middle::ty::{self, Binder, Ty, TyCtxt};
use rustc_span::def_id::LocalDefId;
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits;
use std::cell::RefCell;

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
    can_be_generator: Option<hir::Movability>,
) -> Option<GeneratorTypes<'tcx>> {
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

    if let Some(kind) = body.generator_kind && can_be_generator.is_some() {
        let yield_ty = if kind == hir::GeneratorKind::Gen {
            let yield_ty = fcx
                .next_ty_var(TypeVariableOrigin { kind: TypeVariableOriginKind::TypeInference, span });
            fcx.require_type_is_sized(yield_ty, span, traits::SizedYieldType);
            yield_ty
        } else {
            tcx.mk_unit()
        };

        // Resume type defaults to `()` if the generator has no argument.
        let resume_ty = fn_sig.inputs().get(0).copied().unwrap_or_else(|| tcx.mk_unit());

        fcx.resume_yield_tys = Some((resume_ty, yield_ty));
    }

    GatherLocalsVisitor::new(&fcx).visit_body(body);

    // C-variadic fns also have a `VaList` input that's not listed in `fn_sig`
    // (as it's created inside the body itself, not passed in from outside).
    let maybe_va_list = if fn_sig.c_variadic {
        let span = body.params.last().unwrap().span;
        let va_list_did = tcx.require_lang_item(LangItem::VaList, Some(span));
        let region = fcx.next_region_var(RegionVariableOrigin::MiscVariable(span));

        Some(tcx.bound_type_of(va_list_did).subst(tcx, &[region.into()]))
    } else {
        None
    };

    // Add formal parameters.
    let inputs_hir = hir.fn_decl_by_hir_id(fn_id).map(|decl| &decl.inputs);
    let inputs_fn = fn_sig.inputs().iter().copied();
    for (idx, (param_ty, param)) in inputs_fn.chain(maybe_va_list).zip(body.params).enumerate() {
        // Check the pattern.
        let ty_span = try { inputs_hir?.get(idx)?.span };
        fcx.check_pat_top(&param.pat, param_ty, ty_span, false);

        // Check that argument is Sized.
        // The check for a non-trivial pattern is a hack to avoid duplicate warnings
        // for simple cases like `fn foo(x: Trait)`,
        // where we would error once on the parameter as a whole, and once on the binding `x`.
        if param.pat.simple_ident().is_none() && !tcx.features().unsized_fn_params {
            fcx.require_type_is_sized(param_ty, param.pat.span, traits::SizedArgumentType(ty_span));
        }

        fcx.write_ty(param.hir_id, param_ty);
    }

    fcx.typeck_results.borrow_mut().liberated_fn_sigs_mut().insert(fn_id, fn_sig);

    if let ty::Dynamic(_, _, ty::Dyn) = declared_ret_ty.kind() {
        // FIXME: We need to verify that the return type is `Sized` after the return expression has
        // been evaluated so that we have types available for all the nodes being returned, but that
        // requires the coerced evaluated type to be stored. Moving `check_return_expr` before this
        // causes unsized errors caused by the `declared_ret_ty` to point at the return expression,
        // while keeping the current ordering we will ignore the tail expression's type because we
        // don't know it yet. We can't do `check_expr_kind` while keeping `check_return_expr`
        // because we will trigger "unreachable expression" lints unconditionally.
        // Because of all of this, we perform a crude check to know whether the simplest `!Sized`
        // case that a newcomer might make, returning a bare trait, and in that case we populate
        // the tail expression's type so that the suggestion will be correct, but ignore all other
        // possible cases.
        fcx.check_expr(&body.value);
        fcx.require_type_is_sized(declared_ret_ty, decl.output.span(), traits::SizedReturnType);
    } else {
        fcx.require_type_is_sized(declared_ret_ty, decl.output.span(), traits::SizedReturnType);
        fcx.check_return_expr(&body.value, false);
    }

    // We insert the deferred_generator_interiors entry after visiting the body.
    // This ensures that all nested generators appear before the entry of this generator.
    // resolve_generator_interiors relies on this property.
    let gen_ty = if let (Some(_), Some(gen_kind)) = (can_be_generator, body.generator_kind) {
        let interior = fcx
            .next_ty_var(TypeVariableOrigin { kind: TypeVariableOriginKind::MiscVariable, span });
        fcx.deferred_generator_interiors.borrow_mut().push((
            fn_def_id,
            body.id(),
            interior,
            gen_kind,
        ));

        let (resume_ty, yield_ty) = fcx.resume_yield_tys.unwrap();
        Some(GeneratorTypes {
            resume_ty,
            yield_ty,
            interior,
            movability: can_be_generator.unwrap(),
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
        check_panic_info_fn(tcx, panic_impl_did.expect_local(), fn_sig, decl, declared_ret_ty);
    }

    if let Some(lang_start_defid) = tcx.lang_items().start_fn() && lang_start_defid == fn_def_id.to_def_id() {
        check_lang_start_fn(tcx, fn_sig, decl, fn_def_id);
    }

    gen_ty
}

fn check_panic_info_fn(
    tcx: TyCtxt<'_>,
    fn_id: LocalDefId,
    fn_sig: ty::FnSig<'_>,
    decl: &hir::FnDecl<'_>,
    declared_ret_ty: Ty<'_>,
) {
    let Some(panic_info_did) = tcx.lang_items().panic_info() else {
        tcx.sess.err("language item required, but not found: `panic_info`");
        return;
    };

    if *declared_ret_ty.kind() != ty::Never {
        tcx.sess.span_err(decl.output.span(), "return type should be `!`");
    }

    let inputs = fn_sig.inputs();
    if inputs.len() != 1 {
        tcx.sess.span_err(tcx.def_span(fn_id), "function should have one argument");
        return;
    }

    let arg_is_panic_info = match *inputs[0].kind() {
        ty::Ref(region, ty, mutbl) => match *ty.kind() {
            ty::Adt(ref adt, _) => {
                adt.did() == panic_info_did && mutbl.is_not() && !region.is_static()
            }
            _ => false,
        },
        _ => false,
    };

    if !arg_is_panic_info {
        tcx.sess.span_err(decl.inputs[0].span, "argument should be `&PanicInfo`");
    }

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
}

fn check_lang_start_fn<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_sig: ty::FnSig<'tcx>,
    decl: &'tcx hir::FnDecl<'tcx>,
    def_id: LocalDefId,
) {
    let inputs = fn_sig.inputs();

    let arg_count = inputs.len();
    if arg_count != 4 {
        tcx.sess.emit_err(LangStartIncorrectNumberArgs {
            params_span: tcx.def_span(def_id),
            found_param_count: arg_count,
        });
    }

    // only check args if they should exist by checking the count
    // note: this does not handle args being shifted or their order swapped very nicely
    // but it's a lang item, users shouldn't frequently encounter this

    // first arg is `main: fn() -> T`
    if let Some(&main_arg) = inputs.get(0) {
        // make a Ty for the generic on the fn for diagnostics
        // FIXME: make the lang item generic checks check for the right generic *kind*
        // for example `start`'s generic should be a type parameter
        let generics = tcx.generics_of(def_id);
        let fn_generic = generics.param_at(0, tcx);
        let generic_tykind =
            ty::Param(ty::ParamTy { index: fn_generic.index, name: fn_generic.name });
        let generic_ty = tcx.mk_ty(generic_tykind);
        let expected_fn_sig =
            tcx.mk_fn_sig([].iter(), &generic_ty, false, hir::Unsafety::Normal, Abi::Rust);
        let expected_ty = tcx.mk_fn_ptr(Binder::dummy(expected_fn_sig));

        // we emit the same error to suggest changing the arg no matter what's wrong with the arg
        let emit_main_fn_arg_err = || {
            tcx.sess.emit_err(LangStartIncorrectParam {
                param_span: decl.inputs[0].span,
                param_num: 1,
                expected_ty: expected_ty,
                found_ty: main_arg,
            });
        };

        if let ty::FnPtr(main_fn_sig) = main_arg.kind() {
            let main_fn_inputs = main_fn_sig.inputs();
            if main_fn_inputs.iter().count() != 0 {
                emit_main_fn_arg_err();
            }

            let output = main_fn_sig.output();
            output.map_bound(|ret_ty| {
                // if the output ty is a generic, it's probably the right one
                if !matches!(ret_ty.kind(), ty::Param(_)) {
                    emit_main_fn_arg_err();
                }
            });
        } else {
            emit_main_fn_arg_err();
        }
    }

    // second arg is isize
    if let Some(&argc_arg) = inputs.get(1) {
        if argc_arg != tcx.types.isize {
            tcx.sess.emit_err(LangStartIncorrectParam {
                param_span: decl.inputs[1].span,
                param_num: 2,
                expected_ty: tcx.types.isize,
                found_ty: argc_arg,
            });
        }
    }

    // third arg is `*const *const u8`
    if let Some(&argv_arg) = inputs.get(2) {
        let mut argv_is_okay = false;
        if let ty::RawPtr(outer_ptr) = argv_arg.kind() {
            if outer_ptr.mutbl.is_not() {
                if let ty::RawPtr(inner_ptr) = outer_ptr.ty.kind() {
                    if inner_ptr.mutbl.is_not() && inner_ptr.ty == tcx.types.u8 {
                        argv_is_okay = true;
                    }
                }
            }
        }

        if !argv_is_okay {
            let inner_ptr_ty =
                tcx.mk_ptr(ty::TypeAndMut { mutbl: hir::Mutability::Not, ty: tcx.types.u8 });
            let expected_ty =
                tcx.mk_ptr(ty::TypeAndMut { mutbl: hir::Mutability::Not, ty: inner_ptr_ty });
            tcx.sess.emit_err(LangStartIncorrectParam {
                param_span: decl.inputs[2].span,
                param_num: 3,
                expected_ty,
                found_ty: argv_arg,
            });
        }
    }

    // fourth arg is `sigpipe: u8`
    if let Some(&sigpipe_arg) = inputs.get(3) {
        if sigpipe_arg != tcx.types.u8 {
            tcx.sess.emit_err(LangStartIncorrectParam {
                param_span: decl.inputs[3].span,
                param_num: 4,
                expected_ty: tcx.types.u8,
                found_ty: sigpipe_arg,
            });
        }
    }

    // output type is isize
    if fn_sig.output() != tcx.types.isize {
        tcx.sess.emit_err(LangStartIncorrectRetTy {
            ret_span: decl.output.span(),
            expected_ty: tcx.types.isize,
            found_ty: fn_sig.output(),
        });
    }
}
