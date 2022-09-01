use crate::check::intrinsicck::InlineAsmCtxt;

use super::coercion::CoerceMany;
use super::compare_method::check_type_bounds;
use super::compare_method::{compare_const_impl, compare_impl_method, compare_ty_impl};
use super::*;
use rustc_attr as attr;
use rustc_errors::{Applicability, ErrorGuaranteed, MultiSpan};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::Visitor;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{ItemKind, Node, PathSegment};
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::{DefiningAnchor, RegionVariableOrigin, TyCtxtInferExt};
use rustc_infer::traits::Obligation;
use rustc_lint::builtin::REPR_TRANSPARENT_EXTERNAL_PRIVATE_FIELDS;
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::stability::EvalResult;
use rustc_middle::ty::layout::{LayoutError, MAX_SIMD_LANES};
use rustc_middle::ty::subst::GenericArgKind;
use rustc_middle::ty::util::{Discr, IntTypeExt};
use rustc_middle::ty::{
    self, ParamEnv, ToPredicate, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable,
};
use rustc_session::lint::builtin::{UNINHABITED_STATIC, UNSUPPORTED_CALLING_CONVENTIONS};
use rustc_span::symbol::sym;
use rustc_span::{self, Span};
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits::error_reporting::InferCtxtExt as _;
use rustc_trait_selection::traits::{self, ObligationCtxt};
use rustc_ty_utils::representability::{self, Representability};

use std::ops::ControlFlow;

pub(super) fn check_abi(tcx: TyCtxt<'_>, hir_id: hir::HirId, span: Span, abi: Abi) {
    match tcx.sess.target.is_abi_supported(abi) {
        Some(true) => (),
        Some(false) => {
            struct_span_err!(
                tcx.sess,
                span,
                E0570,
                "`{abi}` is not a supported ABI for the current target",
            )
            .emit();
        }
        None => {
            tcx.struct_span_lint_hir(UNSUPPORTED_CALLING_CONVENTIONS, hir_id, span, |lint| {
                lint.build("use of calling convention not supported on this target").emit();
            });
        }
    }

    // This ABI is only allowed on function pointers
    if abi == Abi::CCmseNonSecureCall {
        struct_span_err!(
            tcx.sess,
            span,
            E0781,
            "the `\"C-cmse-nonsecure-call\"` ABI is only allowed on function pointers"
        )
        .emit();
    }
}

/// Helper used for fns and closures. Does the grungy work of checking a function
/// body and returns the function context used for that purpose, since in the case of a fn item
/// there is still a bit more to do.
///
/// * ...
/// * inherited: other fields inherited from the enclosing fn (if any)
#[instrument(skip(inherited, body), level = "debug")]
pub(super) fn check_fn<'a, 'tcx>(
    inherited: &'a Inherited<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    fn_sig: ty::FnSig<'tcx>,
    decl: &'tcx hir::FnDecl<'tcx>,
    fn_id: hir::HirId,
    body: &'tcx hir::Body<'tcx>,
    can_be_generator: Option<hir::Movability>,
    return_type_pre_known: bool,
) -> (FnCtxt<'a, 'tcx>, Option<GeneratorTypes<'tcx>>) {
    // Create the function context. This is either derived from scratch or,
    // in the case of closures, based on the outer context.
    let mut fcx = FnCtxt::new(inherited, param_env, body.value.hir_id);
    fcx.ps.set(UnsafetyState::function(fn_sig.unsafety, fn_id));
    fcx.return_type_pre_known = return_type_pre_known;

    let tcx = fcx.tcx;
    let hir = tcx.hir();

    let declared_ret_ty = fn_sig.output();

    let ret_ty =
        fcx.register_infer_ok_obligations(fcx.infcx.replace_opaque_types_with_inference_vars(
            declared_ret_ty,
            body.value.hir_id,
            decl.output.span(),
            param_env,
        ));
    // If we replaced declared_ret_ty with infer vars, then we must be inferring
    // an opaque type, so set a flag so we can improve diagnostics.
    fcx.return_type_has_opaque = ret_ty != declared_ret_ty;

    fcx.ret_coercion = Some(RefCell::new(CoerceMany::new(ret_ty)));
    fcx.ret_type_span = Some(decl.output.span());

    let span = body.value.span;

    fn_maybe_err(tcx, span, fn_sig.abi);

    if fn_sig.abi == Abi::RustCall {
        let expected_args = if let ImplicitSelfKind::None = decl.implicit_self { 1 } else { 2 };

        let err = || {
            let item = match tcx.hir().get(fn_id) {
                Node::Item(hir::Item { kind: ItemKind::Fn(header, ..), .. }) => Some(header),
                Node::ImplItem(hir::ImplItem {
                    kind: hir::ImplItemKind::Fn(header, ..), ..
                }) => Some(header),
                Node::TraitItem(hir::TraitItem {
                    kind: hir::TraitItemKind::Fn(header, ..),
                    ..
                }) => Some(header),
                // Closures are RustCall, but they tuple their arguments, so shouldn't be checked
                Node::Expr(hir::Expr { kind: hir::ExprKind::Closure { .. }, .. }) => None,
                node => bug!("Item being checked wasn't a function/closure: {:?}", node),
            };

            if let Some(header) = item {
                tcx.sess.span_err(header.span, "functions with the \"rust-call\" ABI must take a single non-self argument that is a tuple");
            }
        };

        if fn_sig.inputs().len() != expected_args {
            err()
        } else {
            // FIXME(CraftSpider) Add a check on parameter expansion, so we don't just make the ICE happen later on
            //   This will probably require wide-scale changes to support a TupleKind obligation
            //   We can't resolve this without knowing the type of the param
            if !matches!(fn_sig.inputs()[expected_args - 1].kind(), ty::Tuple(_) | ty::Param(_)) {
                err()
            }
        }
    }

    if body.generator_kind.is_some() && can_be_generator.is_some() {
        let yield_ty = fcx
            .next_ty_var(TypeVariableOrigin { kind: TypeVariableOriginKind::TypeInference, span });
        fcx.require_type_is_sized(yield_ty, span, traits::SizedYieldType);

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

    inherited.typeck_results.borrow_mut().liberated_fn_sigs_mut().insert(fn_id, fn_sig);

    fcx.in_tail_expr = true;
    if let ty::Dynamic(..) = declared_ret_ty.kind() {
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
    fcx.in_tail_expr = false;

    // We insert the deferred_generator_interiors entry after visiting the body.
    // This ensures that all nested generators appear before the entry of this generator.
    // resolve_generator_interiors relies on this property.
    let gen_ty = if let (Some(_), Some(gen_kind)) = (can_be_generator, body.generator_kind) {
        let interior = fcx
            .next_ty_var(TypeVariableOrigin { kind: TypeVariableOriginKind::MiscVariable, span });
        fcx.deferred_generator_interiors.borrow_mut().push((body.id(), interior, gen_kind));

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
        && panic_impl_did == hir.local_def_id(fn_id).to_def_id()
    {
        check_panic_info_fn(tcx, panic_impl_did.expect_local(), fn_sig, decl, declared_ret_ty);
    }

    // Check that a function marked as `#[alloc_error_handler]` has signature `fn(Layout) -> !`
    if let Some(alloc_error_handler_did) = tcx.lang_items().oom()
        && alloc_error_handler_did == hir.local_def_id(fn_id).to_def_id()
    {
        check_alloc_error_fn(tcx, alloc_error_handler_did.expect_local(), fn_sig, decl, declared_ret_ty);
    }

    (fcx, gen_ty)
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
                adt.did() == panic_info_did && mutbl == hir::Mutability::Not && !region.is_static()
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

fn check_alloc_error_fn(
    tcx: TyCtxt<'_>,
    fn_id: LocalDefId,
    fn_sig: ty::FnSig<'_>,
    decl: &hir::FnDecl<'_>,
    declared_ret_ty: Ty<'_>,
) {
    let Some(alloc_layout_did) = tcx.lang_items().alloc_layout() else {
        tcx.sess.err("language item required, but not found: `alloc_layout`");
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

    let arg_is_alloc_layout = match inputs[0].kind() {
        ty::Adt(ref adt, _) => adt.did() == alloc_layout_did,
        _ => false,
    };

    if !arg_is_alloc_layout {
        tcx.sess.span_err(decl.inputs[0].span, "argument should be `Layout`");
    }

    let DefKind::Fn = tcx.def_kind(fn_id) else {
        let span = tcx.def_span(fn_id);
        tcx.sess.span_err(span, "`#[alloc_error_handler]` should be a function");
        return;
    };

    let generic_counts = tcx.generics_of(fn_id).own_counts();
    if generic_counts.types != 0 {
        let span = tcx.def_span(fn_id);
        tcx.sess.span_err(span, "`#[alloc_error_handler]` function should have no type parameters");
    }
    if generic_counts.consts != 0 {
        let span = tcx.def_span(fn_id);
        tcx.sess
            .span_err(span, "`#[alloc_error_handler]` function should have no const parameters");
    }
}

fn check_struct(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let def = tcx.adt_def(def_id);
    let span = tcx.def_span(def_id);
    def.destructor(tcx); // force the destructor to be evaluated
    check_representable(tcx, span, def_id);

    if def.repr().simd() {
        check_simd(tcx, span, def_id);
    }

    check_transparent(tcx, span, def);
    check_packed(tcx, span, def);
}

fn check_union(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let def = tcx.adt_def(def_id);
    let span = tcx.def_span(def_id);
    def.destructor(tcx); // force the destructor to be evaluated
    check_representable(tcx, span, def_id);
    check_transparent(tcx, span, def);
    check_union_fields(tcx, span, def_id);
    check_packed(tcx, span, def);
}

/// Check that the fields of the `union` do not need dropping.
fn check_union_fields(tcx: TyCtxt<'_>, span: Span, item_def_id: LocalDefId) -> bool {
    let item_type = tcx.type_of(item_def_id);
    if let ty::Adt(def, substs) = item_type.kind() {
        assert!(def.is_union());

        fn allowed_union_field<'tcx>(
            ty: Ty<'tcx>,
            tcx: TyCtxt<'tcx>,
            param_env: ty::ParamEnv<'tcx>,
            span: Span,
        ) -> bool {
            // We don't just accept all !needs_drop fields, due to semver concerns.
            match ty.kind() {
                ty::Ref(..) => true, // references never drop (even mutable refs, which are non-Copy and hence fail the later check)
                ty::Tuple(tys) => {
                    // allow tuples of allowed types
                    tys.iter().all(|ty| allowed_union_field(ty, tcx, param_env, span))
                }
                ty::Array(elem, _len) => {
                    // Like `Copy`, we do *not* special-case length 0.
                    allowed_union_field(*elem, tcx, param_env, span)
                }
                _ => {
                    // Fallback case: allow `ManuallyDrop` and things that are `Copy`.
                    ty.ty_adt_def().is_some_and(|adt_def| adt_def.is_manually_drop())
                        || ty.is_copy_modulo_regions(tcx.at(span), param_env)
                }
            }
        }

        let param_env = tcx.param_env(item_def_id);
        for field in &def.non_enum_variant().fields {
            let field_ty = field.ty(tcx, substs);

            if !allowed_union_field(field_ty, tcx, param_env, span) {
                let (field_span, ty_span) = match tcx.hir().get_if_local(field.did) {
                    // We are currently checking the type this field came from, so it must be local.
                    Some(Node::Field(field)) => (field.span, field.ty.span),
                    _ => unreachable!("mir field has to correspond to hir field"),
                };
                struct_span_err!(
                    tcx.sess,
                    field_span,
                    E0740,
                    "unions cannot contain fields that may need dropping"
                )
                .note(
                    "a type is guaranteed not to need dropping \
                    when it implements `Copy`, or when it is the special `ManuallyDrop<_>` type",
                )
                .multipart_suggestion_verbose(
                    "when the type does not implement `Copy`, \
                    wrap it inside a `ManuallyDrop<_>` and ensure it is manually dropped",
                    vec![
                        (ty_span.shrink_to_lo(), "std::mem::ManuallyDrop<".into()),
                        (ty_span.shrink_to_hi(), ">".into()),
                    ],
                    Applicability::MaybeIncorrect,
                )
                .emit();
                return false;
            } else if field_ty.needs_drop(tcx, param_env) {
                // This should never happen. But we can get here e.g. in case of name resolution errors.
                tcx.sess.delay_span_bug(span, "we should never accept maybe-dropping union fields");
            }
        }
    } else {
        span_bug!(span, "unions must be ty::Adt, but got {:?}", item_type.kind());
    }
    true
}

/// Check that a `static` is inhabited.
fn check_static_inhabited<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) {
    // Make sure statics are inhabited.
    // Other parts of the compiler assume that there are no uninhabited places. In principle it
    // would be enough to check this for `extern` statics, as statics with an initializer will
    // have UB during initialization if they are uninhabited, but there also seems to be no good
    // reason to allow any statics to be uninhabited.
    let ty = tcx.type_of(def_id);
    let span = tcx.def_span(def_id);
    let layout = match tcx.layout_of(ParamEnv::reveal_all().and(ty)) {
        Ok(l) => l,
        // Foreign statics that overflow their allowed size should emit an error
        Err(LayoutError::SizeOverflow(_))
            if {
                let node = tcx.hir().get_by_def_id(def_id);
                matches!(
                    node,
                    hir::Node::ForeignItem(hir::ForeignItem {
                        kind: hir::ForeignItemKind::Static(..),
                        ..
                    })
                )
            } =>
        {
            tcx.sess
                .struct_span_err(span, "extern static is too large for the current architecture")
                .emit();
            return;
        }
        // Generic statics are rejected, but we still reach this case.
        Err(e) => {
            tcx.sess.delay_span_bug(span, &e.to_string());
            return;
        }
    };
    if layout.abi.is_uninhabited() {
        tcx.struct_span_lint_hir(
            UNINHABITED_STATIC,
            tcx.hir().local_def_id_to_hir_id(def_id),
            span,
            |lint| {
                lint.build("static of uninhabited type")
                .note("uninhabited statics cannot be initialized, and any access would be an immediate error")
                .emit();
            },
        );
    }
}

/// Checks that an opaque type does not contain cycles and does not use `Self` or `T::Foo`
/// projections that would result in "inheriting lifetimes".
pub(super) fn check_opaque<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    substs: SubstsRef<'tcx>,
    origin: &hir::OpaqueTyOrigin,
) {
    let span = tcx.def_span(def_id);
    check_opaque_for_inheriting_lifetimes(tcx, def_id, span);
    if tcx.type_of(def_id).references_error() {
        return;
    }
    if check_opaque_for_cycles(tcx, def_id, substs, span, origin).is_err() {
        return;
    }
    check_opaque_meets_bounds(tcx, def_id, substs, span, origin);
}

/// Checks that an opaque type does not use `Self` or `T::Foo` projections that would result
/// in "inheriting lifetimes".
#[instrument(level = "debug", skip(tcx, span))]
pub(super) fn check_opaque_for_inheriting_lifetimes<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    span: Span,
) {
    let item = tcx.hir().expect_item(def_id);
    debug!(?item, ?span);

    struct FoundParentLifetime;
    struct FindParentLifetimeVisitor<'tcx>(&'tcx ty::Generics);
    impl<'tcx> ty::visit::TypeVisitor<'tcx> for FindParentLifetimeVisitor<'tcx> {
        type BreakTy = FoundParentLifetime;

        fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
            debug!("FindParentLifetimeVisitor: r={:?}", r);
            if let ty::ReEarlyBound(ty::EarlyBoundRegion { index, .. }) = *r {
                if index < self.0.parent_count as u32 {
                    return ControlFlow::Break(FoundParentLifetime);
                } else {
                    return ControlFlow::CONTINUE;
                }
            }

            r.super_visit_with(self)
        }

        fn visit_const(&mut self, c: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
            if let ty::ConstKind::Unevaluated(..) = c.kind() {
                // FIXME(#72219) We currently don't detect lifetimes within substs
                // which would violate this check. Even though the particular substitution is not used
                // within the const, this should still be fixed.
                return ControlFlow::CONTINUE;
            }
            c.super_visit_with(self)
        }
    }

    struct ProhibitOpaqueVisitor<'tcx> {
        tcx: TyCtxt<'tcx>,
        opaque_identity_ty: Ty<'tcx>,
        generics: &'tcx ty::Generics,
        selftys: Vec<(Span, Option<String>)>,
    }

    impl<'tcx> ty::visit::TypeVisitor<'tcx> for ProhibitOpaqueVisitor<'tcx> {
        type BreakTy = Ty<'tcx>;

        fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
            debug!("check_opaque_for_inheriting_lifetimes: (visit_ty) t={:?}", t);
            if t == self.opaque_identity_ty {
                ControlFlow::CONTINUE
            } else {
                t.super_visit_with(&mut FindParentLifetimeVisitor(self.generics))
                    .map_break(|FoundParentLifetime| t)
            }
        }
    }

    impl<'tcx> Visitor<'tcx> for ProhibitOpaqueVisitor<'tcx> {
        type NestedFilter = nested_filter::OnlyBodies;

        fn nested_visit_map(&mut self) -> Self::Map {
            self.tcx.hir()
        }

        fn visit_ty(&mut self, arg: &'tcx hir::Ty<'tcx>) {
            match arg.kind {
                hir::TyKind::Path(hir::QPath::Resolved(None, path)) => match &path.segments {
                    [
                        PathSegment {
                            res: Some(Res::SelfTy { trait_: _, alias_to: impl_ref }),
                            ..
                        },
                    ] => {
                        let impl_ty_name =
                            impl_ref.map(|(def_id, _)| self.tcx.def_path_str(def_id));
                        self.selftys.push((path.span, impl_ty_name));
                    }
                    _ => {}
                },
                _ => {}
            }
            hir::intravisit::walk_ty(self, arg);
        }
    }

    if let ItemKind::OpaqueTy(hir::OpaqueTy {
        origin: hir::OpaqueTyOrigin::AsyncFn(..) | hir::OpaqueTyOrigin::FnReturn(..),
        ..
    }) = item.kind
    {
        let mut visitor = ProhibitOpaqueVisitor {
            opaque_identity_ty: tcx.mk_opaque(
                def_id.to_def_id(),
                InternalSubsts::identity_for_item(tcx, def_id.to_def_id()),
            ),
            generics: tcx.generics_of(def_id),
            tcx,
            selftys: vec![],
        };
        let prohibit_opaque = tcx
            .explicit_item_bounds(def_id)
            .iter()
            .try_for_each(|(predicate, _)| predicate.visit_with(&mut visitor));
        debug!(
            "check_opaque_for_inheriting_lifetimes: prohibit_opaque={:?}, visitor.opaque_identity_ty={:?}, visitor.generics={:?}",
            prohibit_opaque, visitor.opaque_identity_ty, visitor.generics
        );

        if let Some(ty) = prohibit_opaque.break_value() {
            visitor.visit_item(&item);
            let is_async = match item.kind {
                ItemKind::OpaqueTy(hir::OpaqueTy { origin, .. }) => {
                    matches!(origin, hir::OpaqueTyOrigin::AsyncFn(..))
                }
                _ => unreachable!(),
            };

            let mut err = struct_span_err!(
                tcx.sess,
                span,
                E0760,
                "`{}` return type cannot contain a projection or `Self` that references lifetimes from \
                 a parent scope",
                if is_async { "async fn" } else { "impl Trait" },
            );

            for (span, name) in visitor.selftys {
                err.span_suggestion(
                    span,
                    "consider spelling out the type instead",
                    name.unwrap_or_else(|| format!("{:?}", ty)),
                    Applicability::MaybeIncorrect,
                );
            }
            err.emit();
        }
    }
}

/// Checks that an opaque type does not contain cycles.
pub(super) fn check_opaque_for_cycles<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    substs: SubstsRef<'tcx>,
    span: Span,
    origin: &hir::OpaqueTyOrigin,
) -> Result<(), ErrorGuaranteed> {
    if tcx.try_expand_impl_trait_type(def_id.to_def_id(), substs).is_err() {
        let reported = match origin {
            hir::OpaqueTyOrigin::AsyncFn(..) => async_opaque_type_cycle_error(tcx, span),
            _ => opaque_type_cycle_error(tcx, def_id, span),
        };
        Err(reported)
    } else {
        Ok(())
    }
}

/// Check that the concrete type behind `impl Trait` actually implements `Trait`.
///
/// This is mostly checked at the places that specify the opaque type, but we
/// check those cases in the `param_env` of that function, which may have
/// bounds not on this opaque type:
///
/// type X<T> = impl Clone
/// fn f<T: Clone>(t: T) -> X<T> {
///     t
/// }
///
/// Without this check the above code is incorrectly accepted: we would ICE if
/// some tried, for example, to clone an `Option<X<&mut ()>>`.
#[instrument(level = "debug", skip(tcx))]
fn check_opaque_meets_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    substs: SubstsRef<'tcx>,
    span: Span,
    origin: &hir::OpaqueTyOrigin,
) {
    let hidden_type = tcx.bound_type_of(def_id.to_def_id()).subst(tcx, substs);

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let defining_use_anchor = match *origin {
        hir::OpaqueTyOrigin::FnReturn(did) | hir::OpaqueTyOrigin::AsyncFn(did) => did,
        hir::OpaqueTyOrigin::TyAlias => def_id,
    };
    let param_env = tcx.param_env(defining_use_anchor);

    tcx.infer_ctxt().with_opaque_type_inference(DefiningAnchor::Bind(defining_use_anchor)).enter(
        move |infcx| {
            let ocx = ObligationCtxt::new(&infcx);
            let opaque_ty = tcx.mk_opaque(def_id.to_def_id(), substs);

            let misc_cause = traits::ObligationCause::misc(span, hir_id);

            match infcx.at(&misc_cause, param_env).eq(opaque_ty, hidden_type) {
                Ok(infer_ok) => ocx.register_infer_ok_obligations(infer_ok),
                Err(ty_err) => {
                    tcx.sess.delay_span_bug(
                        span,
                        &format!("could not unify `{hidden_type}` with revealed type:\n{ty_err}"),
                    );
                }
            }

            // Additionally require the hidden type to be well-formed with only the generics of the opaque type.
            // Defining use functions may have more bounds than the opaque type, which is ok, as long as the
            // hidden type is well formed even without those bounds.
            let predicate = ty::Binder::dummy(ty::PredicateKind::WellFormed(hidden_type.into()))
                .to_predicate(tcx);
            ocx.register_obligation(Obligation::new(misc_cause, param_env, predicate));

            // Check that all obligations are satisfied by the implementation's
            // version.
            let errors = ocx.select_all_or_error();
            if !errors.is_empty() {
                infcx.report_fulfillment_errors(&errors, None, false);
            }
            match origin {
                // Checked when type checking the function containing them.
                hir::OpaqueTyOrigin::FnReturn(..) | hir::OpaqueTyOrigin::AsyncFn(..) => {}
                // Can have different predicates to their defining use
                hir::OpaqueTyOrigin::TyAlias => {
                    let outlives_environment = OutlivesEnvironment::new(param_env);
                    infcx.check_region_obligations_and_report_errors(
                        defining_use_anchor,
                        &outlives_environment,
                    );
                }
            }
            // Clean up after ourselves
            let _ = infcx.inner.borrow_mut().opaque_type_storage.take_opaque_types();
        },
    );
}

fn check_item_type<'tcx>(tcx: TyCtxt<'tcx>, id: hir::ItemId) {
    debug!(
        "check_item_type(it.def_id={:?}, it.name={})",
        id.def_id,
        tcx.def_path_str(id.def_id.to_def_id())
    );
    let _indenter = indenter();
    match tcx.def_kind(id.def_id) {
        DefKind::Static(..) => {
            tcx.ensure().typeck(id.def_id);
            maybe_check_static_with_link_section(tcx, id.def_id);
            check_static_inhabited(tcx, id.def_id);
        }
        DefKind::Const => {
            tcx.ensure().typeck(id.def_id);
        }
        DefKind::Enum => {
            let item = tcx.hir().item(id);
            let hir::ItemKind::Enum(ref enum_definition, _) = item.kind else {
                return;
            };
            check_enum(tcx, &enum_definition.variants, item.def_id);
        }
        DefKind::Fn => {} // entirely within check_item_body
        DefKind::Impl => {
            let it = tcx.hir().item(id);
            let hir::ItemKind::Impl(ref impl_) = it.kind else {
                return;
            };
            debug!("ItemKind::Impl {} with id {:?}", it.ident, it.def_id);
            if let Some(impl_trait_ref) = tcx.impl_trait_ref(it.def_id) {
                check_impl_items_against_trait(
                    tcx,
                    it.span,
                    it.def_id,
                    impl_trait_ref,
                    &impl_.items,
                );
                check_on_unimplemented(tcx, it);
            }
        }
        DefKind::Trait => {
            let it = tcx.hir().item(id);
            let hir::ItemKind::Trait(_, _, _, _, ref items) = it.kind else {
                return;
            };
            check_on_unimplemented(tcx, it);

            for item in items.iter() {
                let item = tcx.hir().trait_item(item.id);
                match item.kind {
                    hir::TraitItemKind::Fn(ref sig, _) => {
                        let abi = sig.header.abi;
                        fn_maybe_err(tcx, item.ident.span, abi);
                    }
                    hir::TraitItemKind::Type(.., Some(default)) => {
                        let assoc_item = tcx.associated_item(item.def_id);
                        let trait_substs =
                            InternalSubsts::identity_for_item(tcx, it.def_id.to_def_id());
                        let _: Result<_, rustc_errors::ErrorGuaranteed> = check_type_bounds(
                            tcx,
                            assoc_item,
                            assoc_item,
                            default.span,
                            ty::TraitRef { def_id: it.def_id.to_def_id(), substs: trait_substs },
                        );
                    }
                    _ => {}
                }
            }
        }
        DefKind::Struct => {
            check_struct(tcx, id.def_id);
        }
        DefKind::Union => {
            check_union(tcx, id.def_id);
        }
        DefKind::OpaqueTy => {
            let item = tcx.hir().item(id);
            let hir::ItemKind::OpaqueTy(hir::OpaqueTy { origin, .. }) = item.kind else {
                return;
            };
            // HACK(jynelson): trying to infer the type of `impl trait` breaks documenting
            // `async-std` (and `pub async fn` in general).
            // Since rustdoc doesn't care about the concrete type behind `impl Trait`, just don't look at it!
            // See https://github.com/rust-lang/rust/issues/75100
            if !tcx.sess.opts.actually_rustdoc {
                let substs = InternalSubsts::identity_for_item(tcx, item.def_id.to_def_id());
                check_opaque(tcx, item.def_id, substs, &origin);
            }
        }
        DefKind::TyAlias => {
            let pty_ty = tcx.type_of(id.def_id);
            let generics = tcx.generics_of(id.def_id);
            check_type_params_are_used(tcx, &generics, pty_ty);
        }
        DefKind::ForeignMod => {
            let it = tcx.hir().item(id);
            let hir::ItemKind::ForeignMod { abi, items } = it.kind else {
                return;
            };
            check_abi(tcx, it.hir_id(), it.span, abi);

            if abi == Abi::RustIntrinsic {
                for item in items {
                    let item = tcx.hir().foreign_item(item.id);
                    intrinsic::check_intrinsic_type(tcx, item);
                }
            } else if abi == Abi::PlatformIntrinsic {
                for item in items {
                    let item = tcx.hir().foreign_item(item.id);
                    intrinsic::check_platform_intrinsic_type(tcx, item);
                }
            } else {
                for item in items {
                    let def_id = item.id.def_id;
                    let generics = tcx.generics_of(def_id);
                    let own_counts = generics.own_counts();
                    if generics.params.len() - own_counts.lifetimes != 0 {
                        let (kinds, kinds_pl, egs) = match (own_counts.types, own_counts.consts) {
                            (_, 0) => ("type", "types", Some("u32")),
                            // We don't specify an example value, because we can't generate
                            // a valid value for any type.
                            (0, _) => ("const", "consts", None),
                            _ => ("type or const", "types or consts", None),
                        };
                        struct_span_err!(
                            tcx.sess,
                            item.span,
                            E0044,
                            "foreign items may not have {kinds} parameters",
                        )
                        .span_label(item.span, &format!("can't have {kinds} parameters"))
                        .help(
                            // FIXME: once we start storing spans for type arguments, turn this
                            // into a suggestion.
                            &format!(
                                "replace the {} parameters with concrete {}{}",
                                kinds,
                                kinds_pl,
                                egs.map(|egs| format!(" like `{}`", egs)).unwrap_or_default(),
                            ),
                        )
                        .emit();
                    }

                    let item = tcx.hir().foreign_item(item.id);
                    match item.kind {
                        hir::ForeignItemKind::Fn(ref fn_decl, _, _) => {
                            require_c_abi_if_c_variadic(tcx, fn_decl, abi, item.span);
                        }
                        hir::ForeignItemKind::Static(..) => {
                            check_static_inhabited(tcx, def_id);
                        }
                        _ => {}
                    }
                }
            }
        }
        DefKind::GlobalAsm => {
            let it = tcx.hir().item(id);
            let hir::ItemKind::GlobalAsm(asm) = it.kind else { span_bug!(it.span, "DefKind::GlobalAsm but got {:#?}", it) };
            InlineAsmCtxt::new_global_asm(tcx).check_asm(asm, id.hir_id());
        }
        _ => {}
    }
}

pub(super) fn check_on_unimplemented(tcx: TyCtxt<'_>, item: &hir::Item<'_>) {
    // an error would be reported if this fails.
    let _ = traits::OnUnimplementedDirective::of_item(tcx, item.def_id.to_def_id());
}

pub(super) fn check_specialization_validity<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_def: &ty::TraitDef,
    trait_item: &ty::AssocItem,
    impl_id: DefId,
    impl_item: &hir::ImplItemRef,
) {
    let Ok(ancestors) = trait_def.ancestors(tcx, impl_id) else { return };
    let mut ancestor_impls = ancestors.skip(1).filter_map(|parent| {
        if parent.is_from_trait() {
            None
        } else {
            Some((parent, parent.item(tcx, trait_item.def_id)))
        }
    });

    let opt_result = ancestor_impls.find_map(|(parent_impl, parent_item)| {
        match parent_item {
            // Parent impl exists, and contains the parent item we're trying to specialize, but
            // doesn't mark it `default`.
            Some(parent_item) if traits::impl_item_is_final(tcx, &parent_item) => {
                Some(Err(parent_impl.def_id()))
            }

            // Parent impl contains item and makes it specializable.
            Some(_) => Some(Ok(())),

            // Parent impl doesn't mention the item. This means it's inherited from the
            // grandparent. In that case, if parent is a `default impl`, inherited items use the
            // "defaultness" from the grandparent, else they are final.
            None => {
                if tcx.impl_defaultness(parent_impl.def_id()).is_default() {
                    None
                } else {
                    Some(Err(parent_impl.def_id()))
                }
            }
        }
    });

    // If `opt_result` is `None`, we have only encountered `default impl`s that don't contain the
    // item. This is allowed, the item isn't actually getting specialized here.
    let result = opt_result.unwrap_or(Ok(()));

    if let Err(parent_impl) = result {
        report_forbidden_specialization(tcx, impl_item, parent_impl);
    }
}

fn check_impl_items_against_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    full_impl_span: Span,
    impl_id: LocalDefId,
    impl_trait_ref: ty::TraitRef<'tcx>,
    impl_item_refs: &[hir::ImplItemRef],
) {
    // If the trait reference itself is erroneous (so the compilation is going
    // to fail), skip checking the items here -- the `impl_item` table in `tcx`
    // isn't populated for such impls.
    if impl_trait_ref.references_error() {
        return;
    }

    // Negative impls are not expected to have any items
    match tcx.impl_polarity(impl_id) {
        ty::ImplPolarity::Reservation | ty::ImplPolarity::Positive => {}
        ty::ImplPolarity::Negative => {
            if let [first_item_ref, ..] = impl_item_refs {
                let first_item_span = tcx.hir().impl_item(first_item_ref.id).span;
                struct_span_err!(
                    tcx.sess,
                    first_item_span,
                    E0749,
                    "negative impls cannot have any items"
                )
                .emit();
            }
            return;
        }
    }

    let trait_def = tcx.trait_def(impl_trait_ref.def_id);

    for impl_item in impl_item_refs {
        let ty_impl_item = tcx.associated_item(impl_item.id.def_id);
        let ty_trait_item = if let Some(trait_item_id) = ty_impl_item.trait_item_def_id {
            tcx.associated_item(trait_item_id)
        } else {
            // Checked in `associated_item`.
            tcx.sess.delay_span_bug(impl_item.span, "missing associated item in trait");
            continue;
        };
        let impl_item_full = tcx.hir().impl_item(impl_item.id);
        match impl_item_full.kind {
            hir::ImplItemKind::Const(..) => {
                // Find associated const definition.
                compare_const_impl(
                    tcx,
                    &ty_impl_item,
                    impl_item.span,
                    &ty_trait_item,
                    impl_trait_ref,
                );
            }
            hir::ImplItemKind::Fn(..) => {
                let opt_trait_span = tcx.hir().span_if_local(ty_trait_item.def_id);
                compare_impl_method(
                    tcx,
                    &ty_impl_item,
                    &ty_trait_item,
                    impl_trait_ref,
                    opt_trait_span,
                );
            }
            hir::ImplItemKind::TyAlias(impl_ty) => {
                let opt_trait_span = tcx.hir().span_if_local(ty_trait_item.def_id);
                compare_ty_impl(
                    tcx,
                    &ty_impl_item,
                    impl_ty.span,
                    &ty_trait_item,
                    impl_trait_ref,
                    opt_trait_span,
                );
            }
        }

        check_specialization_validity(
            tcx,
            trait_def,
            &ty_trait_item,
            impl_id.to_def_id(),
            impl_item,
        );
    }

    if let Ok(ancestors) = trait_def.ancestors(tcx, impl_id.to_def_id()) {
        // Check for missing items from trait
        let mut missing_items = Vec::new();

        let mut must_implement_one_of: Option<&[Ident]> =
            trait_def.must_implement_one_of.as_deref();

        for &trait_item_id in tcx.associated_item_def_ids(impl_trait_ref.def_id) {
            let is_implemented = ancestors
                .leaf_def(tcx, trait_item_id)
                .map_or(false, |node_item| node_item.item.defaultness(tcx).has_value());

            if !is_implemented && tcx.impl_defaultness(impl_id).is_final() {
                missing_items.push(tcx.associated_item(trait_item_id));
            }

            // true if this item is specifically implemented in this impl
            let is_implemented_here = ancestors
                .leaf_def(tcx, trait_item_id)
                .map_or(false, |node_item| !node_item.defining_node.is_from_trait());

            if !is_implemented_here {
                match tcx.eval_default_body_stability(trait_item_id, full_impl_span) {
                    EvalResult::Deny { feature, reason, issue, .. } => default_body_is_unstable(
                        tcx,
                        full_impl_span,
                        trait_item_id,
                        feature,
                        reason,
                        issue,
                    ),

                    // Unmarked default bodies are considered stable (at least for now).
                    EvalResult::Allow | EvalResult::Unmarked => {}
                }
            }

            if let Some(required_items) = &must_implement_one_of {
                if is_implemented_here {
                    let trait_item = tcx.associated_item(trait_item_id);
                    if required_items.contains(&trait_item.ident(tcx)) {
                        must_implement_one_of = None;
                    }
                }
            }
        }

        if !missing_items.is_empty() {
            missing_items_err(tcx, tcx.def_span(impl_id), &missing_items, full_impl_span);
        }

        if let Some(missing_items) = must_implement_one_of {
            let attr_span = tcx
                .get_attr(impl_trait_ref.def_id, sym::rustc_must_implement_one_of)
                .map(|attr| attr.span);

            missing_items_must_implement_one_of_err(
                tcx,
                tcx.def_span(impl_id),
                missing_items,
                attr_span,
            );
        }
    }
}

/// Checks whether a type can be represented in memory. In particular, it
/// identifies types that contain themselves without indirection through a
/// pointer, which would mean their size is unbounded.
pub(super) fn check_representable(tcx: TyCtxt<'_>, sp: Span, item_def_id: LocalDefId) -> bool {
    let rty = tcx.type_of(item_def_id);

    // Check that it is possible to represent this type. This call identifies
    // (1) types that contain themselves and (2) types that contain a different
    // recursive type. It is only necessary to throw an error on those that
    // contain themselves. For case 2, there must be an inner type that will be
    // caught by case 1.
    match representability::ty_is_representable(tcx, rty, sp, None) {
        Representability::SelfRecursive(spans) => {
            recursive_type_with_infinite_size_error(tcx, item_def_id.to_def_id(), spans);
            return false;
        }
        Representability::Representable | Representability::ContainsRecursive => (),
    }
    true
}

pub fn check_simd(tcx: TyCtxt<'_>, sp: Span, def_id: LocalDefId) {
    let t = tcx.type_of(def_id);
    if let ty::Adt(def, substs) = t.kind()
        && def.is_struct()
    {
        let fields = &def.non_enum_variant().fields;
        if fields.is_empty() {
            struct_span_err!(tcx.sess, sp, E0075, "SIMD vector cannot be empty").emit();
            return;
        }
        let e = fields[0].ty(tcx, substs);
        if !fields.iter().all(|f| f.ty(tcx, substs) == e) {
            struct_span_err!(tcx.sess, sp, E0076, "SIMD vector should be homogeneous")
                .span_label(sp, "SIMD elements must have the same type")
                .emit();
            return;
        }

        let len = if let ty::Array(_ty, c) = e.kind() {
            c.try_eval_usize(tcx, tcx.param_env(def.did()))
        } else {
            Some(fields.len() as u64)
        };
        if let Some(len) = len {
            if len == 0 {
                struct_span_err!(tcx.sess, sp, E0075, "SIMD vector cannot be empty").emit();
                return;
            } else if len > MAX_SIMD_LANES {
                struct_span_err!(
                    tcx.sess,
                    sp,
                    E0075,
                    "SIMD vector cannot have more than {MAX_SIMD_LANES} elements",
                )
                .emit();
                return;
            }
        }

        // Check that we use types valid for use in the lanes of a SIMD "vector register"
        // These are scalar types which directly match a "machine" type
        // Yes: Integers, floats, "thin" pointers
        // No: char, "fat" pointers, compound types
        match e.kind() {
            ty::Param(_) => (), // pass struct<T>(T, T, T, T) through, let monomorphization catch errors
            ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::RawPtr(_) => (), // struct(u8, u8, u8, u8) is ok
            ty::Array(t, _) if matches!(t.kind(), ty::Param(_)) => (), // pass struct<T>([T; N]) through, let monomorphization catch errors
            ty::Array(t, _clen)
                if matches!(
                    t.kind(),
                    ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::RawPtr(_)
                ) =>
            { /* struct([f32; 4]) is ok */ }
            _ => {
                struct_span_err!(
                    tcx.sess,
                    sp,
                    E0077,
                    "SIMD vector element type should be a \
                        primitive scalar (integer/float/pointer) type"
                )
                .emit();
                return;
            }
        }
    }
}

pub(super) fn check_packed(tcx: TyCtxt<'_>, sp: Span, def: ty::AdtDef<'_>) {
    let repr = def.repr();
    if repr.packed() {
        for attr in tcx.get_attrs(def.did(), sym::repr) {
            for r in attr::parse_repr_attr(&tcx.sess, attr) {
                if let attr::ReprPacked(pack) = r
                && let Some(repr_pack) = repr.pack
                && pack as u64 != repr_pack.bytes()
            {
                        struct_span_err!(
                            tcx.sess,
                            sp,
                            E0634,
                            "type has conflicting packed representation hints"
                        )
                        .emit();
            }
            }
        }
        if repr.align.is_some() {
            struct_span_err!(
                tcx.sess,
                sp,
                E0587,
                "type has conflicting packed and align representation hints"
            )
            .emit();
        } else {
            if let Some(def_spans) = check_packed_inner(tcx, def.did(), &mut vec![]) {
                let mut err = struct_span_err!(
                    tcx.sess,
                    sp,
                    E0588,
                    "packed type cannot transitively contain a `#[repr(align)]` type"
                );

                err.span_note(
                    tcx.def_span(def_spans[0].0),
                    &format!(
                        "`{}` has a `#[repr(align)]` attribute",
                        tcx.item_name(def_spans[0].0)
                    ),
                );

                if def_spans.len() > 2 {
                    let mut first = true;
                    for (adt_def, span) in def_spans.iter().skip(1).rev() {
                        let ident = tcx.item_name(*adt_def);
                        err.span_note(
                            *span,
                            &if first {
                                format!(
                                    "`{}` contains a field of type `{}`",
                                    tcx.type_of(def.did()),
                                    ident
                                )
                            } else {
                                format!("...which contains a field of type `{ident}`")
                            },
                        );
                        first = false;
                    }
                }

                err.emit();
            }
        }
    }
}

pub(super) fn check_packed_inner(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    stack: &mut Vec<DefId>,
) -> Option<Vec<(DefId, Span)>> {
    if let ty::Adt(def, substs) = tcx.type_of(def_id).kind() {
        if def.is_struct() || def.is_union() {
            if def.repr().align.is_some() {
                return Some(vec![(def.did(), DUMMY_SP)]);
            }

            stack.push(def_id);
            for field in &def.non_enum_variant().fields {
                if let ty::Adt(def, _) = field.ty(tcx, substs).kind()
                    && !stack.contains(&def.did())
                    && let Some(mut defs) = check_packed_inner(tcx, def.did(), stack)
                {
                    defs.push((def.did(), field.ident(tcx).span));
                    return Some(defs);
                }
            }
            stack.pop();
        }
    }

    None
}

pub(super) fn check_transparent<'tcx>(tcx: TyCtxt<'tcx>, sp: Span, adt: ty::AdtDef<'tcx>) {
    if !adt.repr().transparent() {
        return;
    }

    if adt.is_union() && !tcx.features().transparent_unions {
        feature_err(
            &tcx.sess.parse_sess,
            sym::transparent_unions,
            sp,
            "transparent unions are unstable",
        )
        .emit();
    }

    if adt.variants().len() != 1 {
        bad_variant_count(tcx, adt, sp, adt.did());
        if adt.variants().is_empty() {
            // Don't bother checking the fields. No variants (and thus no fields) exist.
            return;
        }
    }

    // For each field, figure out if it's known to be a ZST and align(1), with "known"
    // respecting #[non_exhaustive] attributes.
    let field_infos = adt.all_fields().map(|field| {
        let ty = field.ty(tcx, InternalSubsts::identity_for_item(tcx, field.did));
        let param_env = tcx.param_env(field.did);
        let layout = tcx.layout_of(param_env.and(ty));
        // We are currently checking the type this field came from, so it must be local
        let span = tcx.hir().span_if_local(field.did).unwrap();
        let zst = layout.map_or(false, |layout| layout.is_zst());
        let align1 = layout.map_or(false, |layout| layout.align.abi.bytes() == 1);
        if !zst {
            return (span, zst, align1, None);
        }

        fn check_non_exhaustive<'tcx>(
            tcx: TyCtxt<'tcx>,
            t: Ty<'tcx>,
        ) -> ControlFlow<(&'static str, DefId, SubstsRef<'tcx>, bool)> {
            match t.kind() {
                ty::Tuple(list) => list.iter().try_for_each(|t| check_non_exhaustive(tcx, t)),
                ty::Array(ty, _) => check_non_exhaustive(tcx, *ty),
                ty::Adt(def, subst) => {
                    if !def.did().is_local() {
                        let non_exhaustive = def.is_variant_list_non_exhaustive()
                            || def
                                .variants()
                                .iter()
                                .any(ty::VariantDef::is_field_list_non_exhaustive);
                        let has_priv = def.all_fields().any(|f| !f.vis.is_public());
                        if non_exhaustive || has_priv {
                            return ControlFlow::Break((
                                def.descr(),
                                def.did(),
                                subst,
                                non_exhaustive,
                            ));
                        }
                    }
                    def.all_fields()
                        .map(|field| field.ty(tcx, subst))
                        .try_for_each(|t| check_non_exhaustive(tcx, t))
                }
                _ => ControlFlow::Continue(()),
            }
        }

        (span, zst, align1, check_non_exhaustive(tcx, ty).break_value())
    });

    let non_zst_fields = field_infos
        .clone()
        .filter_map(|(span, zst, _align1, _non_exhaustive)| if !zst { Some(span) } else { None });
    let non_zst_count = non_zst_fields.clone().count();
    if non_zst_count >= 2 {
        bad_non_zero_sized_fields(tcx, adt, non_zst_count, non_zst_fields, sp);
    }
    let incompatible_zst_fields =
        field_infos.clone().filter(|(_, _, _, opt)| opt.is_some()).count();
    let incompat = incompatible_zst_fields + non_zst_count >= 2 && non_zst_count < 2;
    for (span, zst, align1, non_exhaustive) in field_infos {
        if zst && !align1 {
            struct_span_err!(
                tcx.sess,
                span,
                E0691,
                "zero-sized field in transparent {} has alignment larger than 1",
                adt.descr(),
            )
            .span_label(span, "has alignment larger than 1")
            .emit();
        }
        if incompat && let Some((descr, def_id, substs, non_exhaustive)) = non_exhaustive {
            tcx.struct_span_lint_hir(
                REPR_TRANSPARENT_EXTERNAL_PRIVATE_FIELDS,
                tcx.hir().local_def_id_to_hir_id(adt.did().expect_local()),
                span,
                |lint| {
                    let note = if non_exhaustive {
                        "is marked with `#[non_exhaustive]`"
                    } else {
                        "contains private fields"
                    };
                    let field_ty = tcx.def_path_str_with_substs(def_id, substs);
                    lint.build("zero-sized fields in repr(transparent) cannot contain external non-exhaustive types")
                        .note(format!("this {descr} contains `{field_ty}`, which {note}, \
                            and makes it not a breaking change to become non-zero-sized in the future."))
                        .emit();
                },
            )
        }
    }
}

#[allow(trivial_numeric_casts)]
fn check_enum<'tcx>(tcx: TyCtxt<'tcx>, vs: &'tcx [hir::Variant<'tcx>], def_id: LocalDefId) {
    let def = tcx.adt_def(def_id);
    let sp = tcx.def_span(def_id);
    def.destructor(tcx); // force the destructor to be evaluated

    if vs.is_empty() {
        if let Some(attr) = tcx.get_attr(def_id.to_def_id(), sym::repr) {
            struct_span_err!(
                tcx.sess,
                attr.span,
                E0084,
                "unsupported representation for zero-variant enum"
            )
            .span_label(sp, "zero-variant enum")
            .emit();
        }
    }

    let repr_type_ty = def.repr().discr_type().to_ty(tcx);
    if repr_type_ty == tcx.types.i128 || repr_type_ty == tcx.types.u128 {
        if !tcx.features().repr128 {
            feature_err(
                &tcx.sess.parse_sess,
                sym::repr128,
                sp,
                "repr with 128-bit type is unstable",
            )
            .emit();
        }
    }

    for v in vs {
        if let Some(ref e) = v.disr_expr {
            tcx.ensure().typeck(tcx.hir().local_def_id(e.hir_id));
        }
    }

    if tcx.adt_def(def_id).repr().int.is_none() && tcx.features().arbitrary_enum_discriminant {
        let is_unit = |var: &hir::Variant<'_>| matches!(var.data, hir::VariantData::Unit(..));

        let has_disr = |var: &hir::Variant<'_>| var.disr_expr.is_some();
        let has_non_units = vs.iter().any(|var| !is_unit(var));
        let disr_units = vs.iter().any(|var| is_unit(&var) && has_disr(&var));
        let disr_non_unit = vs.iter().any(|var| !is_unit(&var) && has_disr(&var));

        if disr_non_unit || (disr_units && has_non_units) {
            let mut err =
                struct_span_err!(tcx.sess, sp, E0732, "`#[repr(inttype)]` must be specified");
            err.emit();
        }
    }

    detect_discriminant_duplicate(tcx, def.discriminants(tcx).collect(), vs, sp);

    check_representable(tcx, sp, def_id);
    check_transparent(tcx, sp, def);
}

/// Part of enum check. Given the discriminants of an enum, errors if two or more discriminants are equal
fn detect_discriminant_duplicate<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut discrs: Vec<(VariantIdx, Discr<'tcx>)>,
    vs: &'tcx [hir::Variant<'tcx>],
    self_span: Span,
) {
    // Helper closure to reduce duplicate code. This gets called everytime we detect a duplicate.
    // Here `idx` refers to the order of which the discriminant appears, and its index in `vs`
    let report = |dis: Discr<'tcx>, idx: usize, err: &mut Diagnostic| {
        let var = &vs[idx]; // HIR for the duplicate discriminant
        let (span, display_discr) = match var.disr_expr {
            Some(ref expr) => {
                // In the case the discriminant is both a duplicate and overflowed, let the user know
                if let hir::ExprKind::Lit(lit) = &tcx.hir().body(expr.body).value.kind
                    && let rustc_ast::LitKind::Int(lit_value, _int_kind) = &lit.node
                    && *lit_value != dis.val
                {
                    (tcx.hir().span(expr.hir_id), format!("`{dis}` (overflowed from `{lit_value}`)"))
                // Otherwise, format the value as-is
                } else {
                    (tcx.hir().span(expr.hir_id), format!("`{dis}`"))
                }
            }
            None => {
                // At this point we know this discriminant is a duplicate, and was not explicitly
                // assigned by the user. Here we iterate backwards to fetch the HIR for the last
                // explicitly assigned discriminant, and letting the user know that this was the
                // increment startpoint, and how many steps from there leading to the duplicate
                if let Some((n, hir::Variant { span, ident, .. })) =
                    vs[..idx].iter().rev().enumerate().find(|v| v.1.disr_expr.is_some())
                {
                    let ve_ident = var.ident;
                    let n = n + 1;
                    let sp = if n > 1 { "variants" } else { "variant" };

                    err.span_label(
                        *span,
                        format!("discriminant for `{ve_ident}` incremented from this startpoint (`{ident}` + {n} {sp} later => `{ve_ident}` = {dis})"),
                    );
                }

                (vs[idx].span, format!("`{dis}`"))
            }
        };

        err.span_label(span, format!("{display_discr} assigned here"));
    };

    // Here we loop through the discriminants, comparing each discriminant to another.
    // When a duplicate is detected, we instantiate an error and point to both
    // initial and duplicate value. The duplicate discriminant is then discarded by swapping
    // it with the last element and decrementing the `vec.len` (which is why we have to evaluate
    // `discrs.len()` anew every iteration, and why this could be tricky to do in a functional
    // style as we are mutating `discrs` on the fly).
    let mut i = 0;
    while i < discrs.len() {
        let hir_var_i_idx = discrs[i].0.index();
        let mut error: Option<DiagnosticBuilder<'_, _>> = None;

        let mut o = i + 1;
        while o < discrs.len() {
            let hir_var_o_idx = discrs[o].0.index();

            if discrs[i].1.val == discrs[o].1.val {
                let err = error.get_or_insert_with(|| {
                    let mut ret = struct_span_err!(
                        tcx.sess,
                        self_span,
                        E0081,
                        "discriminant value `{}` assigned more than once",
                        discrs[i].1,
                    );

                    report(discrs[i].1, hir_var_i_idx, &mut ret);

                    ret
                });

                report(discrs[o].1, hir_var_o_idx, err);

                // Safe to unwrap here, as we wouldn't reach this point if `discrs` was empty
                discrs[o] = *discrs.last().unwrap();
                discrs.pop();
            } else {
                o += 1;
            }
        }

        if let Some(mut e) = error {
            e.emit();
        }

        i += 1;
    }
}

pub(super) fn check_type_params_are_used<'tcx>(
    tcx: TyCtxt<'tcx>,
    generics: &ty::Generics,
    ty: Ty<'tcx>,
) {
    debug!("check_type_params_are_used(generics={:?}, ty={:?})", generics, ty);

    assert_eq!(generics.parent, None);

    if generics.own_counts().types == 0 {
        return;
    }

    let mut params_used = BitSet::new_empty(generics.params.len());

    if ty.references_error() {
        // If there is already another error, do not emit
        // an error for not using a type parameter.
        assert!(tcx.sess.has_errors().is_some());
        return;
    }

    for leaf in ty.walk() {
        if let GenericArgKind::Type(leaf_ty) = leaf.unpack()
            && let ty::Param(param) = leaf_ty.kind()
        {
            debug!("found use of ty param {:?}", param);
            params_used.insert(param.index);
        }
    }

    for param in &generics.params {
        if !params_used.contains(param.index)
            && let ty::GenericParamDefKind::Type { .. } = param.kind
        {
            let span = tcx.def_span(param.def_id);
            struct_span_err!(
                tcx.sess,
                span,
                E0091,
                "type parameter `{}` is unused",
                param.name,
            )
            .span_label(span, "unused type parameter")
            .emit();
        }
    }
}

pub(super) fn check_mod_item_types(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    let module = tcx.hir_module_items(module_def_id);
    for id in module.items() {
        check_item_type(tcx, id);
    }
}

fn async_opaque_type_cycle_error(tcx: TyCtxt<'_>, span: Span) -> ErrorGuaranteed {
    struct_span_err!(tcx.sess, span, E0733, "recursion in an `async fn` requires boxing")
        .span_label(span, "recursive `async fn`")
        .note("a recursive `async fn` must be rewritten to return a boxed `dyn Future`")
        .note(
            "consider using the `async_recursion` crate: https://crates.io/crates/async_recursion",
        )
        .emit()
}

/// Emit an error for recursive opaque types.
///
/// If this is a return `impl Trait`, find the item's return expressions and point at them. For
/// direct recursion this is enough, but for indirect recursion also point at the last intermediary
/// `impl Trait`.
///
/// If all the return expressions evaluate to `!`, then we explain that the error will go away
/// after changing it. This can happen when a user uses `panic!()` or similar as a placeholder.
fn opaque_type_cycle_error(tcx: TyCtxt<'_>, def_id: LocalDefId, span: Span) -> ErrorGuaranteed {
    let mut err = struct_span_err!(tcx.sess, span, E0720, "cannot resolve opaque type");

    let mut label = false;
    if let Some((def_id, visitor)) = get_owner_return_paths(tcx, def_id) {
        let typeck_results = tcx.typeck(def_id);
        if visitor
            .returns
            .iter()
            .filter_map(|expr| typeck_results.node_type_opt(expr.hir_id))
            .all(|ty| matches!(ty.kind(), ty::Never))
        {
            let spans = visitor
                .returns
                .iter()
                .filter(|expr| typeck_results.node_type_opt(expr.hir_id).is_some())
                .map(|expr| expr.span)
                .collect::<Vec<Span>>();
            let span_len = spans.len();
            if span_len == 1 {
                err.span_label(spans[0], "this returned value is of `!` type");
            } else {
                let mut multispan: MultiSpan = spans.clone().into();
                for span in spans {
                    multispan.push_span_label(span, "this returned value is of `!` type");
                }
                err.span_note(multispan, "these returned values have a concrete \"never\" type");
            }
            err.help("this error will resolve once the item's body returns a concrete type");
        } else {
            let mut seen = FxHashSet::default();
            seen.insert(span);
            err.span_label(span, "recursive opaque type");
            label = true;
            for (sp, ty) in visitor
                .returns
                .iter()
                .filter_map(|e| typeck_results.node_type_opt(e.hir_id).map(|t| (e.span, t)))
                .filter(|(_, ty)| !matches!(ty.kind(), ty::Never))
            {
                struct OpaqueTypeCollector(Vec<DefId>);
                impl<'tcx> ty::visit::TypeVisitor<'tcx> for OpaqueTypeCollector {
                    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
                        match *t.kind() {
                            ty::Opaque(def, _) => {
                                self.0.push(def);
                                ControlFlow::CONTINUE
                            }
                            _ => t.super_visit_with(self),
                        }
                    }
                }
                let mut visitor = OpaqueTypeCollector(vec![]);
                ty.visit_with(&mut visitor);
                for def_id in visitor.0 {
                    let ty_span = tcx.def_span(def_id);
                    if !seen.contains(&ty_span) {
                        err.span_label(ty_span, &format!("returning this opaque type `{ty}`"));
                        seen.insert(ty_span);
                    }
                    err.span_label(sp, &format!("returning here with type `{ty}`"));
                }
            }
        }
    }
    if !label {
        err.span_label(span, "cannot resolve opaque type");
    }
    err.emit()
}
