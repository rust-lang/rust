use super::coercion::CoerceMany;
use super::compare_method::check_type_bounds;
use super::compare_method::{compare_const_impl, compare_impl_method, compare_ty_impl};
use super::*;

use rustc_attr as attr;
use rustc_errors::{Applicability, ErrorReported};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId, LOCAL_CRATE};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{ItemKind, Node};
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::{RegionVariableOrigin, TyCtxtInferExt};
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::subst::GenericArgKind;
use rustc_middle::ty::util::{Discr, IntTypeExt, Representability};
use rustc_middle::ty::{self, ParamEnv, RegionKind, ToPredicate, Ty, TyCtxt};
use rustc_session::config::EntryFnType;
use rustc_session::lint::builtin::UNINHABITED_STATIC;
use rustc_span::symbol::sym;
use rustc_span::{self, MultiSpan, Span};
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::opaque_types::InferCtxtExt as _;
use rustc_trait_selection::traits::error_reporting::InferCtxtExt as _;
use rustc_trait_selection::traits::{self, ObligationCauseCode};

use std::ops::ControlFlow;

pub fn check_wf_new(tcx: TyCtxt<'_>) {
    let visit = wfcheck::CheckTypeWellFormedVisitor::new(tcx);
    tcx.hir().krate().par_visit_all_item_likes(&visit);
}

pub(super) fn check_abi(tcx: TyCtxt<'_>, span: Span, abi: Abi) {
    if !tcx.sess.target.is_abi_supported(abi) {
        struct_span_err!(
            tcx.sess,
            span,
            E0570,
            "The ABI `{}` is not supported for the current target",
            abi
        )
        .emit()
    }
}

/// Helper used for fns and closures. Does the grungy work of checking a function
/// body and returns the function context used for that purpose, since in the case of a fn item
/// there is still a bit more to do.
///
/// * ...
/// * inherited: other fields inherited from the enclosing fn (if any)
pub(super) fn check_fn<'a, 'tcx>(
    inherited: &'a Inherited<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    fn_sig: ty::FnSig<'tcx>,
    decl: &'tcx hir::FnDecl<'tcx>,
    fn_id: hir::HirId,
    body: &'tcx hir::Body<'tcx>,
    can_be_generator: Option<hir::Movability>,
) -> (FnCtxt<'a, 'tcx>, Option<GeneratorTypes<'tcx>>) {
    let mut fn_sig = fn_sig;

    debug!("check_fn(sig={:?}, fn_id={}, param_env={:?})", fn_sig, fn_id, param_env);

    // Create the function context. This is either derived from scratch or,
    // in the case of closures, based on the outer context.
    let mut fcx = FnCtxt::new(inherited, param_env, body.value.hir_id);
    *fcx.ps.borrow_mut() = UnsafetyState::function(fn_sig.unsafety, fn_id);

    let tcx = fcx.tcx;
    let sess = tcx.sess;
    let hir = tcx.hir();

    let declared_ret_ty = fn_sig.output();

    let revealed_ret_ty =
        fcx.instantiate_opaque_types_from_value(fn_id, declared_ret_ty, decl.output.span());
    debug!("check_fn: declared_ret_ty: {}, revealed_ret_ty: {}", declared_ret_ty, revealed_ret_ty);
    fcx.ret_coercion = Some(RefCell::new(CoerceMany::new(revealed_ret_ty)));
    fcx.ret_type_span = Some(decl.output.span());
    if let ty::Opaque(..) = declared_ret_ty.kind() {
        fcx.ret_coercion_impl_trait = Some(declared_ret_ty);
    }
    fn_sig = tcx.mk_fn_sig(
        fn_sig.inputs().iter().cloned(),
        revealed_ret_ty,
        fn_sig.c_variadic,
        fn_sig.unsafety,
        fn_sig.abi,
    );

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
                Node::Expr(hir::Expr { kind: hir::ExprKind::Closure(..), .. }) => None,
                node => bug!("Item being checked wasn't a function/closure: {:?}", node),
            };

            if let Some(header) = item {
                tcx.sess.span_err(header.span, "A function with the \"rust-call\" ABI must take a single non-self argument that is a tuple")
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

    let outer_def_id = tcx.closure_base_def_id(hir.local_def_id(fn_id).to_def_id()).expect_local();
    let outer_hir_id = hir.local_def_id_to_hir_id(outer_def_id);
    GatherLocalsVisitor::new(&fcx, outer_hir_id).visit_body(body);

    // C-variadic fns also have a `VaList` input that's not listed in `fn_sig`
    // (as it's created inside the body itself, not passed in from outside).
    let maybe_va_list = if fn_sig.c_variadic {
        let span = body.params.last().unwrap().span;
        let va_list_did = tcx.require_lang_item(LangItem::VaList, Some(span));
        let region = fcx.next_region_var(RegionVariableOrigin::MiscVariable(span));

        Some(tcx.type_of(va_list_did).subst(tcx, &[region.into()]))
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
        tcx.sess.delay_span_bug(decl.output.span(), "`!Sized` return type");
    } else {
        fcx.require_type_is_sized(declared_ret_ty, decl.output.span(), traits::SizedReturnType);
        fcx.check_return_expr(&body.value);
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
    //
    // However, the behavior around `!` is sort of complex. In the
    // event that the `actual_return_ty` comes back as `!`, that
    // indicates that the fn either does not return or "returns" only
    // values of type `!`. In this case, if there is an expected
    // return type that is *not* `!`, that should be ok. But if the
    // return type is being inferred, we want to "fallback" to `!`:
    //
    //     let x = move || panic!();
    //
    // To allow for that, I am creating a type variable with diverging
    // fallback. This was deemed ever so slightly better than unifying
    // the return value with `!` because it allows for the caller to
    // make more assumptions about the return type (e.g., they could do
    //
    //     let y: Option<u32> = Some(x());
    //
    // which would then cause this return type to become `u32`, not
    // `!`).
    let coercion = fcx.ret_coercion.take().unwrap().into_inner();
    let mut actual_return_ty = coercion.complete(&fcx);
    if actual_return_ty.is_never() {
        actual_return_ty = fcx.next_diverging_ty_var(TypeVariableOrigin {
            kind: TypeVariableOriginKind::DivergingFn,
            span,
        });
    }
    fcx.demand_suptype(span, revealed_ret_ty, actual_return_ty);

    // Check that the main return type implements the termination trait.
    if let Some(term_id) = tcx.lang_items().termination() {
        if let Some((def_id, EntryFnType::Main)) = tcx.entry_fn(LOCAL_CRATE) {
            let main_id = hir.local_def_id_to_hir_id(def_id);
            if main_id == fn_id {
                let substs = tcx.mk_substs_trait(declared_ret_ty, &[]);
                let trait_ref = ty::TraitRef::new(term_id, substs);
                let return_ty_span = decl.output.span();
                let cause = traits::ObligationCause::new(
                    return_ty_span,
                    fn_id,
                    ObligationCauseCode::MainFunctionType,
                );

                inherited.register_predicate(traits::Obligation::new(
                    cause,
                    param_env,
                    trait_ref.without_const().to_predicate(tcx),
                ));
            }
        }
    }

    // Check that a function marked as `#[panic_handler]` has signature `fn(&PanicInfo) -> !`
    if let Some(panic_impl_did) = tcx.lang_items().panic_impl() {
        if panic_impl_did == hir.local_def_id(fn_id).to_def_id() {
            if let Some(panic_info_did) = tcx.lang_items().panic_info() {
                if *declared_ret_ty.kind() != ty::Never {
                    sess.span_err(decl.output.span(), "return type should be `!`");
                }

                let inputs = fn_sig.inputs();
                let span = hir.span(fn_id);
                if inputs.len() == 1 {
                    let arg_is_panic_info = match *inputs[0].kind() {
                        ty::Ref(region, ty, mutbl) => match *ty.kind() {
                            ty::Adt(ref adt, _) => {
                                adt.did == panic_info_did
                                    && mutbl == hir::Mutability::Not
                                    && *region != RegionKind::ReStatic
                            }
                            _ => false,
                        },
                        _ => false,
                    };

                    if !arg_is_panic_info {
                        sess.span_err(decl.inputs[0].span, "argument should be `&PanicInfo`");
                    }

                    if let Node::Item(item) = hir.get(fn_id) {
                        if let ItemKind::Fn(_, ref generics, _) = item.kind {
                            if !generics.params.is_empty() {
                                sess.span_err(span, "should have no type parameters");
                            }
                        }
                    }
                } else {
                    let span = sess.source_map().guess_head_span(span);
                    sess.span_err(span, "function should have one argument");
                }
            } else {
                sess.err("language item required, but not found: `panic_info`");
            }
        }
    }

    // Check that a function marked as `#[alloc_error_handler]` has signature `fn(Layout) -> !`
    if let Some(alloc_error_handler_did) = tcx.lang_items().oom() {
        if alloc_error_handler_did == hir.local_def_id(fn_id).to_def_id() {
            if let Some(alloc_layout_did) = tcx.lang_items().alloc_layout() {
                if *declared_ret_ty.kind() != ty::Never {
                    sess.span_err(decl.output.span(), "return type should be `!`");
                }

                let inputs = fn_sig.inputs();
                let span = hir.span(fn_id);
                if inputs.len() == 1 {
                    let arg_is_alloc_layout = match inputs[0].kind() {
                        ty::Adt(ref adt, _) => adt.did == alloc_layout_did,
                        _ => false,
                    };

                    if !arg_is_alloc_layout {
                        sess.span_err(decl.inputs[0].span, "argument should be `Layout`");
                    }

                    if let Node::Item(item) = hir.get(fn_id) {
                        if let ItemKind::Fn(_, ref generics, _) = item.kind {
                            if !generics.params.is_empty() {
                                sess.span_err(
                                    span,
                                    "`#[alloc_error_handler]` function should have no type \
                                     parameters",
                                );
                            }
                        }
                    }
                } else {
                    let span = sess.source_map().guess_head_span(span);
                    sess.span_err(span, "function should have one argument");
                }
            } else {
                sess.err("language item required, but not found: `alloc_layout`");
            }
        }
    }

    (fcx, gen_ty)
}

pub(super) fn check_struct(tcx: TyCtxt<'_>, id: hir::HirId, span: Span) {
    let def_id = tcx.hir().local_def_id(id);
    let def = tcx.adt_def(def_id);
    def.destructor(tcx); // force the destructor to be evaluated
    check_representable(tcx, span, def_id);

    if def.repr.simd() {
        check_simd(tcx, span, def_id);
    }

    check_transparent(tcx, span, def);
    check_packed(tcx, span, def);
}

fn check_union(tcx: TyCtxt<'_>, id: hir::HirId, span: Span) {
    let def_id = tcx.hir().local_def_id(id);
    let def = tcx.adt_def(def_id);
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
        let fields = &def.non_enum_variant().fields;
        let param_env = tcx.param_env(item_def_id);
        for field in fields {
            let field_ty = field.ty(tcx, substs);
            // We are currently checking the type this field came from, so it must be local.
            let field_span = tcx.hir().span_if_local(field.did).unwrap();
            if field_ty.needs_drop(tcx, param_env) {
                struct_span_err!(
                    tcx.sess,
                    field_span,
                    E0740,
                    "unions may not contain fields that need dropping"
                )
                .span_note(field_span, "`std::mem::ManuallyDrop` can be used to wrap the type")
                .emit();
                return false;
            }
        }
    } else {
        span_bug!(span, "unions must be ty::Adt, but got {:?}", item_type.kind());
    }
    true
}

/// Check that a `static` is inhabited.
fn check_static_inhabited<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId, span: Span) {
    // Make sure statics are inhabited.
    // Other parts of the compiler assume that there are no uninhabited places. In principle it
    // would be enough to check this for `extern` statics, as statics with an initializer will
    // have UB during initialization if they are uninhabited, but there also seems to be no good
    // reason to allow any statics to be uninhabited.
    let ty = tcx.type_of(def_id);
    let layout = match tcx.layout_of(ParamEnv::reveal_all().and(ty)) {
        Ok(l) => l,
        Err(_) => {
            // Generic statics are rejected, but we still reach this case.
            tcx.sess.delay_span_bug(span, "generic static must be rejected");
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
    span: Span,
    origin: &hir::OpaqueTyOrigin,
) {
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
#[instrument(skip(tcx, span))]
pub(super) fn check_opaque_for_inheriting_lifetimes(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    span: Span,
) {
    let item = tcx.hir().expect_item(tcx.hir().local_def_id_to_hir_id(def_id));
    debug!(?item, ?span);

    struct FoundParentLifetime;
    struct FindParentLifetimeVisitor<'tcx>(&'tcx ty::Generics);
    impl<'tcx> ty::fold::TypeVisitor<'tcx> for FindParentLifetimeVisitor<'tcx> {
        type BreakTy = FoundParentLifetime;

        fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
            debug!("FindParentLifetimeVisitor: r={:?}", r);
            if let RegionKind::ReEarlyBound(ty::EarlyBoundRegion { index, .. }) = r {
                if *index < self.0.parent_count as u32 {
                    return ControlFlow::Break(FoundParentLifetime);
                } else {
                    return ControlFlow::CONTINUE;
                }
            }

            r.super_visit_with(self)
        }

        fn visit_const(&mut self, c: &'tcx ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
            if let ty::ConstKind::Unevaluated(..) = c.val {
                // FIXME(#72219) We currently don't detect lifetimes within substs
                // which would violate this check. Even though the particular substitution is not used
                // within the const, this should still be fixed.
                return ControlFlow::CONTINUE;
            }
            c.super_visit_with(self)
        }
    }

    #[derive(Debug)]
    struct ProhibitOpaqueVisitor<'tcx> {
        opaque_identity_ty: Ty<'tcx>,
        generics: &'tcx ty::Generics,
    }

    impl<'tcx> ty::fold::TypeVisitor<'tcx> for ProhibitOpaqueVisitor<'tcx> {
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

    if let ItemKind::OpaqueTy(hir::OpaqueTy {
        origin: hir::OpaqueTyOrigin::AsyncFn | hir::OpaqueTyOrigin::FnReturn,
        ..
    }) = item.kind
    {
        let mut visitor = ProhibitOpaqueVisitor {
            opaque_identity_ty: tcx.mk_opaque(
                def_id.to_def_id(),
                InternalSubsts::identity_for_item(tcx, def_id.to_def_id()),
            ),
            generics: tcx.generics_of(def_id),
        };
        let prohibit_opaque = tcx
            .explicit_item_bounds(def_id)
            .iter()
            .try_for_each(|(predicate, _)| predicate.visit_with(&mut visitor));
        debug!(
            "check_opaque_for_inheriting_lifetimes: prohibit_opaque={:?}, visitor={:?}",
            prohibit_opaque, visitor
        );

        if let Some(ty) = prohibit_opaque.break_value() {
            let is_async = match item.kind {
                ItemKind::OpaqueTy(hir::OpaqueTy { origin, .. }) => {
                    matches!(origin, hir::OpaqueTyOrigin::AsyncFn)
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

            if let Ok(snippet) = tcx.sess.source_map().span_to_snippet(span) {
                if snippet == "Self" {
                    err.span_suggestion(
                        span,
                        "consider spelling out the type instead",
                        format!("{:?}", ty),
                        Applicability::MaybeIncorrect,
                    );
                }
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
) -> Result<(), ErrorReported> {
    if let Err(partially_expanded_type) = tcx.try_expand_impl_trait_type(def_id.to_def_id(), substs)
    {
        match origin {
            hir::OpaqueTyOrigin::AsyncFn => async_opaque_type_cycle_error(tcx, span),
            hir::OpaqueTyOrigin::Binding => {
                binding_opaque_type_cycle_error(tcx, def_id, span, partially_expanded_type)
            }
            _ => opaque_type_cycle_error(tcx, def_id, span),
        }
        Err(ErrorReported)
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
fn check_opaque_meets_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    substs: SubstsRef<'tcx>,
    span: Span,
    origin: &hir::OpaqueTyOrigin,
) {
    match origin {
        // Checked when type checking the function containing them.
        hir::OpaqueTyOrigin::FnReturn | hir::OpaqueTyOrigin::AsyncFn => return,
        // Can have different predicates to their defining use
        hir::OpaqueTyOrigin::Binding | hir::OpaqueTyOrigin::Misc => {}
    }

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let param_env = tcx.param_env(def_id);

    tcx.infer_ctxt().enter(move |infcx| {
        let inh = Inherited::new(infcx, def_id);
        let infcx = &inh.infcx;
        let opaque_ty = tcx.mk_opaque(def_id.to_def_id(), substs);

        let misc_cause = traits::ObligationCause::misc(span, hir_id);

        let (_, opaque_type_map) = inh.register_infer_ok_obligations(
            infcx.instantiate_opaque_types(def_id, hir_id, param_env, opaque_ty, span),
        );

        for (def_id, opaque_defn) in opaque_type_map {
            match infcx
                .at(&misc_cause, param_env)
                .eq(opaque_defn.concrete_ty, tcx.type_of(def_id).subst(tcx, opaque_defn.substs))
            {
                Ok(infer_ok) => inh.register_infer_ok_obligations(infer_ok),
                Err(ty_err) => tcx.sess.delay_span_bug(
                    opaque_defn.definition_span,
                    &format!(
                        "could not unify `{}` with revealed type:\n{}",
                        opaque_defn.concrete_ty, ty_err,
                    ),
                ),
            }
        }

        // Check that all obligations are satisfied by the implementation's
        // version.
        if let Err(ref errors) = inh.fulfillment_cx.borrow_mut().select_all_or_error(&infcx) {
            infcx.report_fulfillment_errors(errors, None, false);
        }

        // Finally, resolve all regions. This catches wily misuses of
        // lifetime parameters.
        let fcx = FnCtxt::new(&inh, param_env, hir_id);
        fcx.regionck_item(hir_id, span, &[]);
    });
}

pub fn check_item_type<'tcx>(tcx: TyCtxt<'tcx>, it: &'tcx hir::Item<'tcx>) {
    debug!(
        "check_item_type(it.hir_id={}, it.name={})",
        it.hir_id,
        tcx.def_path_str(tcx.hir().local_def_id(it.hir_id).to_def_id())
    );
    let _indenter = indenter();
    match it.kind {
        // Consts can play a role in type-checking, so they are included here.
        hir::ItemKind::Static(..) => {
            let def_id = tcx.hir().local_def_id(it.hir_id);
            tcx.ensure().typeck(def_id);
            maybe_check_static_with_link_section(tcx, def_id, it.span);
            check_static_inhabited(tcx, def_id, it.span);
        }
        hir::ItemKind::Const(..) => {
            tcx.ensure().typeck(tcx.hir().local_def_id(it.hir_id));
        }
        hir::ItemKind::Enum(ref enum_definition, _) => {
            check_enum(tcx, it.span, &enum_definition.variants, it.hir_id);
        }
        hir::ItemKind::Fn(..) => {} // entirely within check_item_body
        hir::ItemKind::Impl { ref items, .. } => {
            debug!("ItemKind::Impl {} with id {}", it.ident, it.hir_id);
            let impl_def_id = tcx.hir().local_def_id(it.hir_id);
            if let Some(impl_trait_ref) = tcx.impl_trait_ref(impl_def_id) {
                check_impl_items_against_trait(tcx, it.span, impl_def_id, impl_trait_ref, items);
                let trait_def_id = impl_trait_ref.def_id;
                check_on_unimplemented(tcx, trait_def_id, it);
            }
        }
        hir::ItemKind::Trait(_, _, _, _, ref items) => {
            let def_id = tcx.hir().local_def_id(it.hir_id);
            check_on_unimplemented(tcx, def_id.to_def_id(), it);

            for item in items.iter() {
                let item = tcx.hir().trait_item(item.id);
                match item.kind {
                    hir::TraitItemKind::Fn(ref sig, _) => {
                        let abi = sig.header.abi;
                        fn_maybe_err(tcx, item.ident.span, abi);
                    }
                    hir::TraitItemKind::Type(.., Some(_default)) => {
                        let item_def_id = tcx.hir().local_def_id(item.hir_id).to_def_id();
                        let assoc_item = tcx.associated_item(item_def_id);
                        let trait_substs =
                            InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                        let _: Result<_, rustc_errors::ErrorReported> = check_type_bounds(
                            tcx,
                            assoc_item,
                            assoc_item,
                            item.span,
                            ty::TraitRef { def_id: def_id.to_def_id(), substs: trait_substs },
                        );
                    }
                    _ => {}
                }
            }
        }
        hir::ItemKind::Struct(..) => {
            check_struct(tcx, it.hir_id, it.span);
        }
        hir::ItemKind::Union(..) => {
            check_union(tcx, it.hir_id, it.span);
        }
        hir::ItemKind::OpaqueTy(hir::OpaqueTy { origin, .. }) => {
            // HACK(jynelson): trying to infer the type of `impl trait` breaks documenting
            // `async-std` (and `pub async fn` in general).
            // Since rustdoc doesn't care about the concrete type behind `impl Trait`, just don't look at it!
            // See https://github.com/rust-lang/rust/issues/75100
            if !tcx.sess.opts.actually_rustdoc {
                let def_id = tcx.hir().local_def_id(it.hir_id);

                let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                check_opaque(tcx, def_id, substs, it.span, &origin);
            }
        }
        hir::ItemKind::TyAlias(..) => {
            let def_id = tcx.hir().local_def_id(it.hir_id);
            let pty_ty = tcx.type_of(def_id);
            let generics = tcx.generics_of(def_id);
            check_type_params_are_used(tcx, &generics, pty_ty);
        }
        hir::ItemKind::ForeignMod { abi, items } => {
            check_abi(tcx, it.span, abi);

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
                    let def_id = tcx.hir().local_def_id(item.id.hir_id);
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
                            "foreign items may not have {} parameters",
                            kinds,
                        )
                        .span_label(item.span, &format!("can't have {} parameters", kinds))
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
                            check_static_inhabited(tcx, def_id, item.span);
                        }
                        _ => {}
                    }
                }
            }
        }
        _ => { /* nothing to do */ }
    }
}

pub(super) fn check_on_unimplemented(tcx: TyCtxt<'_>, trait_def_id: DefId, item: &hir::Item<'_>) {
    let item_def_id = tcx.hir().local_def_id(item.hir_id);
    // an error would be reported if this fails.
    let _ = traits::OnUnimplementedDirective::of_item(tcx, trait_def_id, item_def_id.to_def_id());
}

pub(super) fn check_specialization_validity<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_def: &ty::TraitDef,
    trait_item: &ty::AssocItem,
    impl_id: DefId,
    impl_item: &hir::ImplItem<'_>,
) {
    let kind = match impl_item.kind {
        hir::ImplItemKind::Const(..) => ty::AssocKind::Const,
        hir::ImplItemKind::Fn(..) => ty::AssocKind::Fn,
        hir::ImplItemKind::TyAlias(_) => ty::AssocKind::Type,
    };

    let ancestors = match trait_def.ancestors(tcx, impl_id) {
        Ok(ancestors) => ancestors,
        Err(_) => return,
    };
    let mut ancestor_impls = ancestors
        .skip(1)
        .filter_map(|parent| {
            if parent.is_from_trait() {
                None
            } else {
                Some((parent, parent.item(tcx, trait_item.ident, kind, trait_def.def_id)))
            }
        })
        .peekable();

    if ancestor_impls.peek().is_none() {
        // No parent, nothing to specialize.
        return;
    }

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

pub(super) fn check_impl_items_against_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    full_impl_span: Span,
    impl_id: LocalDefId,
    impl_trait_ref: ty::TraitRef<'tcx>,
    impl_item_refs: &[hir::ImplItemRef<'_>],
) {
    let impl_span = tcx.sess.source_map().guess_head_span(full_impl_span);

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

    // Locate trait definition and items
    let trait_def = tcx.trait_def(impl_trait_ref.def_id);

    let impl_items = || impl_item_refs.iter().map(|iiref| tcx.hir().impl_item(iiref.id));

    // Check existing impl methods to see if they are both present in trait
    // and compatible with trait signature
    for impl_item in impl_items() {
        let namespace = impl_item.kind.namespace();
        let ty_impl_item = tcx.associated_item(tcx.hir().local_def_id(impl_item.hir_id));
        let ty_trait_item = tcx
            .associated_items(impl_trait_ref.def_id)
            .find_by_name_and_namespace(tcx, ty_impl_item.ident, namespace, impl_trait_ref.def_id)
            .or_else(|| {
                // Not compatible, but needed for the error message
                tcx.associated_items(impl_trait_ref.def_id)
                    .filter_by_name(tcx, ty_impl_item.ident, impl_trait_ref.def_id)
                    .next()
            });

        // Check that impl definition matches trait definition
        if let Some(ty_trait_item) = ty_trait_item {
            match impl_item.kind {
                hir::ImplItemKind::Const(..) => {
                    // Find associated const definition.
                    if ty_trait_item.kind == ty::AssocKind::Const {
                        compare_const_impl(
                            tcx,
                            &ty_impl_item,
                            impl_item.span,
                            &ty_trait_item,
                            impl_trait_ref,
                        );
                    } else {
                        let mut err = struct_span_err!(
                            tcx.sess,
                            impl_item.span,
                            E0323,
                            "item `{}` is an associated const, \
                             which doesn't match its trait `{}`",
                            ty_impl_item.ident,
                            impl_trait_ref.print_only_trait_path()
                        );
                        err.span_label(impl_item.span, "does not match trait");
                        // We can only get the spans from local trait definition
                        // Same for E0324 and E0325
                        if let Some(trait_span) = tcx.hir().span_if_local(ty_trait_item.def_id) {
                            err.span_label(trait_span, "item in trait");
                        }
                        err.emit()
                    }
                }
                hir::ImplItemKind::Fn(..) => {
                    let opt_trait_span = tcx.hir().span_if_local(ty_trait_item.def_id);
                    if ty_trait_item.kind == ty::AssocKind::Fn {
                        compare_impl_method(
                            tcx,
                            &ty_impl_item,
                            impl_item.span,
                            &ty_trait_item,
                            impl_trait_ref,
                            opt_trait_span,
                        );
                    } else {
                        let mut err = struct_span_err!(
                            tcx.sess,
                            impl_item.span,
                            E0324,
                            "item `{}` is an associated method, \
                             which doesn't match its trait `{}`",
                            ty_impl_item.ident,
                            impl_trait_ref.print_only_trait_path()
                        );
                        err.span_label(impl_item.span, "does not match trait");
                        if let Some(trait_span) = opt_trait_span {
                            err.span_label(trait_span, "item in trait");
                        }
                        err.emit()
                    }
                }
                hir::ImplItemKind::TyAlias(_) => {
                    let opt_trait_span = tcx.hir().span_if_local(ty_trait_item.def_id);
                    if ty_trait_item.kind == ty::AssocKind::Type {
                        compare_ty_impl(
                            tcx,
                            &ty_impl_item,
                            impl_item.span,
                            &ty_trait_item,
                            impl_trait_ref,
                            opt_trait_span,
                        );
                    } else {
                        let mut err = struct_span_err!(
                            tcx.sess,
                            impl_item.span,
                            E0325,
                            "item `{}` is an associated type, \
                             which doesn't match its trait `{}`",
                            ty_impl_item.ident,
                            impl_trait_ref.print_only_trait_path()
                        );
                        err.span_label(impl_item.span, "does not match trait");
                        if let Some(trait_span) = opt_trait_span {
                            err.span_label(trait_span, "item in trait");
                        }
                        err.emit()
                    }
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
    }

    // Check for missing items from trait
    let mut missing_items = Vec::new();
    if let Ok(ancestors) = trait_def.ancestors(tcx, impl_id.to_def_id()) {
        for trait_item in tcx.associated_items(impl_trait_ref.def_id).in_definition_order() {
            let is_implemented = ancestors
                .leaf_def(tcx, trait_item.ident, trait_item.kind)
                .map(|node_item| !node_item.defining_node.is_from_trait())
                .unwrap_or(false);

            if !is_implemented && tcx.impl_defaultness(impl_id).is_final() {
                if !trait_item.defaultness.has_value() {
                    missing_items.push(*trait_item);
                }
            }
        }
    }

    if !missing_items.is_empty() {
        missing_items_err(tcx, impl_span, &missing_items, full_impl_span);
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
    match rty.is_representable(tcx, sp) {
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
    if let ty::Adt(def, substs) = t.kind() {
        if def.is_struct() {
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
            match e.kind() {
                ty::Param(_) => { /* struct<T>(T, T, T, T) is ok */ }
                _ if e.is_machine() => { /* struct(u8, u8, u8, u8) is ok */ }
                ty::Array(ty, _c) if ty.is_machine() => { /* struct([f32; 4]) */ }
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
}

pub(super) fn check_packed(tcx: TyCtxt<'_>, sp: Span, def: &ty::AdtDef) {
    let repr = def.repr;
    if repr.packed() {
        for attr in tcx.get_attrs(def.did).iter() {
            for r in attr::find_repr_attrs(&tcx.sess, attr) {
                if let attr::ReprPacked(pack) = r {
                    if let Some(repr_pack) = repr.pack {
                        if pack as u64 != repr_pack.bytes() {
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
            if let Some(def_spans) = check_packed_inner(tcx, def.did, &mut vec![]) {
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
                                    tcx.type_of(def.did),
                                    ident
                                )
                            } else {
                                format!("...which contains a field of type `{}`", ident)
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
            if def.repr.align.is_some() {
                return Some(vec![(def.did, DUMMY_SP)]);
            }

            stack.push(def_id);
            for field in &def.non_enum_variant().fields {
                if let ty::Adt(def, _) = field.ty(tcx, substs).kind() {
                    if !stack.contains(&def.did) {
                        if let Some(mut defs) = check_packed_inner(tcx, def.did, stack) {
                            defs.push((def.did, field.ident.span));
                            return Some(defs);
                        }
                    }
                }
            }
            stack.pop();
        }
    }

    None
}

pub(super) fn check_transparent<'tcx>(tcx: TyCtxt<'tcx>, sp: Span, adt: &'tcx ty::AdtDef) {
    if !adt.repr.transparent() {
        return;
    }
    let sp = tcx.sess.source_map().guess_head_span(sp);

    if adt.is_union() && !tcx.features().transparent_unions {
        feature_err(
            &tcx.sess.parse_sess,
            sym::transparent_unions,
            sp,
            "transparent unions are unstable",
        )
        .emit();
    }

    if adt.variants.len() != 1 {
        bad_variant_count(tcx, adt, sp, adt.did);
        if adt.variants.is_empty() {
            // Don't bother checking the fields. No variants (and thus no fields) exist.
            return;
        }
    }

    // For each field, figure out if it's known to be a ZST and align(1)
    let field_infos = adt.all_fields().map(|field| {
        let ty = field.ty(tcx, InternalSubsts::identity_for_item(tcx, field.did));
        let param_env = tcx.param_env(field.did);
        let layout = tcx.layout_of(param_env.and(ty));
        // We are currently checking the type this field came from, so it must be local
        let span = tcx.hir().span_if_local(field.did).unwrap();
        let zst = layout.map(|layout| layout.is_zst()).unwrap_or(false);
        let align1 = layout.map(|layout| layout.align.abi.bytes() == 1).unwrap_or(false);
        (span, zst, align1)
    });

    let non_zst_fields =
        field_infos.clone().filter_map(|(span, zst, _align1)| if !zst { Some(span) } else { None });
    let non_zst_count = non_zst_fields.clone().count();
    if non_zst_count != 1 {
        bad_non_zero_sized_fields(tcx, adt, non_zst_count, non_zst_fields, sp);
    }
    for (span, zst, align1) in field_infos {
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
    }
}

#[allow(trivial_numeric_casts)]
pub fn check_enum<'tcx>(
    tcx: TyCtxt<'tcx>,
    sp: Span,
    vs: &'tcx [hir::Variant<'tcx>],
    id: hir::HirId,
) {
    let def_id = tcx.hir().local_def_id(id);
    let def = tcx.adt_def(def_id);
    def.destructor(tcx); // force the destructor to be evaluated

    if vs.is_empty() {
        let attributes = tcx.get_attrs(def_id.to_def_id());
        if let Some(attr) = tcx.sess.find_by_name(&attributes, sym::repr) {
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

    let repr_type_ty = def.repr.discr_type().to_ty(tcx);
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

    if tcx.adt_def(def_id).repr.int.is_none() && tcx.features().arbitrary_enum_discriminant {
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

    let mut disr_vals: Vec<Discr<'tcx>> = Vec::with_capacity(vs.len());
    for ((_, discr), v) in def.discriminants(tcx).zip(vs) {
        // Check for duplicate discriminant values
        if let Some(i) = disr_vals.iter().position(|&x| x.val == discr.val) {
            let variant_did = def.variants[VariantIdx::new(i)].def_id;
            let variant_i_hir_id = tcx.hir().local_def_id_to_hir_id(variant_did.expect_local());
            let variant_i = tcx.hir().expect_variant(variant_i_hir_id);
            let i_span = match variant_i.disr_expr {
                Some(ref expr) => tcx.hir().span(expr.hir_id),
                None => tcx.hir().span(variant_i_hir_id),
            };
            let span = match v.disr_expr {
                Some(ref expr) => tcx.hir().span(expr.hir_id),
                None => v.span,
            };
            struct_span_err!(
                tcx.sess,
                span,
                E0081,
                "discriminant value `{}` already exists",
                disr_vals[i]
            )
            .span_label(i_span, format!("first use of `{}`", disr_vals[i]))
            .span_label(span, format!("enum already has `{}`", disr_vals[i]))
            .emit();
        }
        disr_vals.push(discr);
    }

    check_representable(tcx, sp, def_id);
    check_transparent(tcx, sp, def);
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
        assert!(tcx.sess.has_errors());
        return;
    }

    for leaf in ty.walk() {
        if let GenericArgKind::Type(leaf_ty) = leaf.unpack() {
            if let ty::Param(param) = leaf_ty.kind() {
                debug!("found use of ty param {:?}", param);
                params_used.insert(param.index);
            }
        }
    }

    for param in &generics.params {
        if !params_used.contains(param.index) {
            if let ty::GenericParamDefKind::Type { .. } = param.kind {
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
}

pub(super) fn check_mod_item_types(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut CheckItemTypesVisitor { tcx });
}

pub(super) fn check_item_well_formed(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    wfcheck::check_item_well_formed(tcx, def_id);
}

pub(super) fn check_trait_item_well_formed(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    wfcheck::check_trait_item(tcx, def_id);
}

pub(super) fn check_impl_item_well_formed(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    wfcheck::check_impl_item(tcx, def_id);
}

fn async_opaque_type_cycle_error(tcx: TyCtxt<'tcx>, span: Span) {
    struct_span_err!(tcx.sess, span, E0733, "recursion in an `async fn` requires boxing")
        .span_label(span, "recursive `async fn`")
        .note("a recursive `async fn` must be rewritten to return a boxed `dyn Future`")
        .emit();
}

/// Emit an error for recursive opaque types.
///
/// If this is a return `impl Trait`, find the item's return expressions and point at them. For
/// direct recursion this is enough, but for indirect recursion also point at the last intermediary
/// `impl Trait`.
///
/// If all the return expressions evaluate to `!`, then we explain that the error will go away
/// after changing it. This can happen when a user uses `panic!()` or similar as a placeholder.
fn opaque_type_cycle_error(tcx: TyCtxt<'tcx>, def_id: LocalDefId, span: Span) {
    let mut err = struct_span_err!(tcx.sess, span, E0720, "cannot resolve opaque type");

    let mut label = false;
    if let Some((hir_id, visitor)) = get_owner_return_paths(tcx, def_id) {
        let typeck_results = tcx.typeck(tcx.hir().local_def_id(hir_id));
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
                    multispan
                        .push_span_label(span, "this returned value is of `!` type".to_string());
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
                struct VisitTypes(Vec<DefId>);
                impl<'tcx> ty::fold::TypeVisitor<'tcx> for VisitTypes {
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
                let mut visitor = VisitTypes(vec![]);
                ty.visit_with(&mut visitor);
                for def_id in visitor.0 {
                    let ty_span = tcx.def_span(def_id);
                    if !seen.contains(&ty_span) {
                        err.span_label(ty_span, &format!("returning this opaque type `{}`", ty));
                        seen.insert(ty_span);
                    }
                    err.span_label(sp, &format!("returning here with type `{}`", ty));
                }
            }
        }
    }
    if !label {
        err.span_label(span, "cannot resolve opaque type");
    }
    err.emit();
}
