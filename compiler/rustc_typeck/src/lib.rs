/*!

# typeck

The type checker is responsible for:

1. Determining the type of each expression.
2. Resolving methods and traits.
3. Guaranteeing that most type rules are met. ("Most?", you say, "why most?"
   Well, dear reader, read on.)

The main entry point is [`check_crate()`]. Type checking operates in
several major phases:

1. The collect phase first passes over all items and determines their
   type, without examining their "innards".

2. Variance inference then runs to compute the variance of each parameter.

3. Coherence checks for overlapping or orphaned impls.

4. Finally, the check phase then checks function bodies and so forth.
   Within the check phase, we check each function body one at a time
   (bodies of function expressions are checked as part of the
   containing function).  Inference is used to supply types wherever
   they are unknown. The actual checking of a function itself has
   several phases (check, regionck, writeback), as discussed in the
   documentation for the [`check`] module.

The type checker is defined into various submodules which are documented
independently:

- astconv: converts the AST representation of types
  into the `ty` representation.

- collect: computes the types of each top-level item and enters them into
  the `tcx.types` table for later use.

- coherence: enforces coherence rules, builds some tables.

- variance: variance inference

- outlives: outlives inference

- check: walks over function bodies and type checks them, inferring types for
  local variables, type parameters, etc as necessary.

- infer: finds the types to use for each type variable such that
  all subtyping and assignment constraints are met.  In essence, the check
  module specifies the constraints, and the infer module solves them.

## Note

This API is completely unstable and subject to change.

*/

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(bool_to_option)]
#![feature(crate_visibility_modifier)]
#![feature(if_let_guard)]
#![feature(is_sorted)]
#![feature(let_else)]
#![feature(min_specialization)]
#![feature(nll)]
#![feature(try_blocks)]
#![feature(never_type)]
#![feature(slice_partition_dedup)]
#![feature(control_flow_enum)]
#![feature(hash_drain_filter)]
#![recursion_limit = "256"]
#![cfg_attr(not(bootstrap), allow(rustc::potential_query_instability))]

#[macro_use]
extern crate tracing;

#[macro_use]
extern crate rustc_middle;

// These are used by Clippy.
pub mod check;
pub mod expr_use_visitor;

mod astconv;
mod bounds;
mod check_unused;
mod coherence;
mod collect;
mod constrained_generic_params;
mod errors;
pub mod hir_wf_check;
mod impl_wf_check;
mod mem_categorization;
mod outlives;
mod structured_errors;
mod variance;

use rustc_errors::{struct_span_err, ErrorReported};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::{Node, CRATE_HIR_ID};
use rustc_infer::infer::{InferOk, TyCtxtInferExt};
use rustc_infer::traits::TraitEngineExt as _;
use rustc_middle::middle;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::util;
use rustc_session::config::EntryFnType;
use rustc_span::{symbol::sym, Span, DUMMY_SP};
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::error_reporting::InferCtxtExt as _;
use rustc_trait_selection::traits::{
    self, ObligationCause, ObligationCauseCode, TraitEngine, TraitEngineExt as _,
};

use std::iter;

use astconv::AstConv;
use bounds::Bounds;

fn require_c_abi_if_c_variadic(tcx: TyCtxt<'_>, decl: &hir::FnDecl<'_>, abi: Abi, span: Span) {
    match (decl.c_variadic, abi) {
        // The function has the correct calling convention, or isn't a "C-variadic" function.
        (false, _) | (true, Abi::C { .. }) | (true, Abi::Cdecl { .. }) => {}
        // The function is a "C-variadic" function with an incorrect calling convention.
        (true, _) => {
            let mut err = struct_span_err!(
                tcx.sess,
                span,
                E0045,
                "C-variadic function must have C or cdecl calling convention"
            );
            err.span_label(span, "C-variadics require C or cdecl calling convention").emit();
        }
    }
}

fn require_same_types<'tcx>(
    tcx: TyCtxt<'tcx>,
    cause: &ObligationCause<'tcx>,
    expected: Ty<'tcx>,
    actual: Ty<'tcx>,
) -> bool {
    tcx.infer_ctxt().enter(|ref infcx| {
        let param_env = ty::ParamEnv::empty();
        let mut fulfill_cx = <dyn TraitEngine<'_>>::new(infcx.tcx);
        match infcx.at(cause, param_env).eq(expected, actual) {
            Ok(InferOk { obligations, .. }) => {
                fulfill_cx.register_predicate_obligations(infcx, obligations);
            }
            Err(err) => {
                infcx.report_mismatched_types(cause, expected, actual, err).emit();
                return false;
            }
        }

        match fulfill_cx.select_all_or_error(infcx).as_slice() {
            [] => true,
            errors => {
                infcx.report_fulfillment_errors(errors, None, false);
                false
            }
        }
    })
}

fn check_main_fn_ty(tcx: TyCtxt<'_>, main_def_id: DefId) {
    let main_fnsig = tcx.fn_sig(main_def_id);
    let main_span = tcx.def_span(main_def_id);

    fn main_fn_diagnostics_hir_id(tcx: TyCtxt<'_>, def_id: DefId, sp: Span) -> hir::HirId {
        if let Some(local_def_id) = def_id.as_local() {
            let hir_id = tcx.hir().local_def_id_to_hir_id(local_def_id);
            let hir_type = tcx.type_of(local_def_id);
            if !matches!(hir_type.kind(), ty::FnDef(..)) {
                span_bug!(sp, "main has a non-function type: found `{}`", hir_type);
            }
            hir_id
        } else {
            CRATE_HIR_ID
        }
    }

    fn main_fn_generics_params_span(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Span> {
        if !def_id.is_local() {
            return None;
        }
        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
        match tcx.hir().find(hir_id) {
            Some(Node::Item(hir::Item { kind: hir::ItemKind::Fn(_, ref generics, _), .. })) => {
                if !generics.params.is_empty() {
                    Some(generics.span)
                } else {
                    None
                }
            }
            _ => {
                span_bug!(tcx.def_span(def_id), "main has a non-function type");
            }
        }
    }

    fn main_fn_where_clauses_span(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Span> {
        if !def_id.is_local() {
            return None;
        }
        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
        match tcx.hir().find(hir_id) {
            Some(Node::Item(hir::Item { kind: hir::ItemKind::Fn(_, ref generics, _), .. })) => {
                generics.where_clause.span()
            }
            _ => {
                span_bug!(tcx.def_span(def_id), "main has a non-function type");
            }
        }
    }

    fn main_fn_asyncness_span(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Span> {
        if !def_id.is_local() {
            return None;
        }
        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
        match tcx.hir().find(hir_id) {
            Some(Node::Item(hir::Item { span: item_span, .. })) => {
                Some(tcx.sess.source_map().guess_head_span(*item_span))
            }
            _ => {
                span_bug!(tcx.def_span(def_id), "main has a non-function type");
            }
        }
    }

    fn main_fn_return_type_span(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Span> {
        if !def_id.is_local() {
            return None;
        }
        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
        match tcx.hir().find(hir_id) {
            Some(Node::Item(hir::Item { kind: hir::ItemKind::Fn(ref fn_sig, _, _), .. })) => {
                Some(fn_sig.decl.output.span())
            }
            _ => {
                span_bug!(tcx.def_span(def_id), "main has a non-function type");
            }
        }
    }

    let mut error = false;
    let main_diagnostics_hir_id = main_fn_diagnostics_hir_id(tcx, main_def_id, main_span);
    let main_fn_generics = tcx.generics_of(main_def_id);
    let main_fn_predicates = tcx.predicates_of(main_def_id);
    if main_fn_generics.count() != 0 || !main_fnsig.bound_vars().is_empty() {
        let generics_param_span = main_fn_generics_params_span(tcx, main_def_id);
        let msg = "`main` function is not allowed to have generic \
            parameters";
        let mut diag =
            struct_span_err!(tcx.sess, generics_param_span.unwrap_or(main_span), E0131, "{}", msg);
        if let Some(generics_param_span) = generics_param_span {
            let label = "`main` cannot have generic parameters".to_string();
            diag.span_label(generics_param_span, label);
        }
        diag.emit();
        error = true;
    } else if !main_fn_predicates.predicates.is_empty() {
        // generics may bring in implicit predicates, so we skip this check if generics is present.
        let generics_where_clauses_span = main_fn_where_clauses_span(tcx, main_def_id);
        let mut diag = struct_span_err!(
            tcx.sess,
            generics_where_clauses_span.unwrap_or(main_span),
            E0646,
            "`main` function is not allowed to have a `where` clause"
        );
        if let Some(generics_where_clauses_span) = generics_where_clauses_span {
            diag.span_label(generics_where_clauses_span, "`main` cannot have a `where` clause");
        }
        diag.emit();
        error = true;
    }

    let main_asyncness = tcx.asyncness(main_def_id);
    if let hir::IsAsync::Async = main_asyncness {
        let mut diag = struct_span_err!(
            tcx.sess,
            main_span,
            E0752,
            "`main` function is not allowed to be `async`"
        );
        let asyncness_span = main_fn_asyncness_span(tcx, main_def_id);
        if let Some(asyncness_span) = asyncness_span {
            diag.span_label(asyncness_span, "`main` function is not allowed to be `async`");
        }
        diag.emit();
        error = true;
    }

    for attr in tcx.get_attrs(main_def_id) {
        if attr.has_name(sym::track_caller) {
            tcx.sess
                .struct_span_err(
                    attr.span,
                    "`main` function is not allowed to be `#[track_caller]`",
                )
                .span_label(main_span, "`main` function is not allowed to be `#[track_caller]`")
                .emit();
            error = true;
        }
    }

    if error {
        return;
    }

    let expected_return_type;
    if let Some(term_id) = tcx.lang_items().termination() {
        let return_ty = main_fnsig.output();
        let return_ty_span = main_fn_return_type_span(tcx, main_def_id).unwrap_or(main_span);
        if !return_ty.bound_vars().is_empty() {
            let msg = "`main` function return type is not allowed to have generic \
                    parameters"
                .to_owned();
            struct_span_err!(tcx.sess, return_ty_span, E0131, "{}", msg).emit();
            error = true;
        }
        let return_ty = return_ty.skip_binder();
        tcx.infer_ctxt().enter(|infcx| {
            let cause = traits::ObligationCause::new(
                return_ty_span,
                main_diagnostics_hir_id,
                ObligationCauseCode::MainFunctionType,
            );
            let mut fulfillment_cx = traits::FulfillmentContext::new();
            // normalize any potential projections in the return type, then add
            // any possible obligations to the fulfillment context.
            // HACK(ThePuzzlemaker) this feels symptomatic of a problem within
            // checking trait fulfillment, not this here. I'm not sure why it
            // works in the example in `fn test()` given in #88609? This also
            // probably isn't the best way to do this.
            let InferOk { value: norm_return_ty, obligations } = infcx
                .partially_normalize_associated_types_in(
                    cause.clone(),
                    ty::ParamEnv::empty(),
                    return_ty,
                );
            fulfillment_cx.register_predicate_obligations(&infcx, obligations);
            fulfillment_cx.register_bound(
                &infcx,
                ty::ParamEnv::empty(),
                norm_return_ty,
                term_id,
                cause,
            );
            let errors = fulfillment_cx.select_all_or_error(&infcx);
            if !errors.is_empty() {
                infcx.report_fulfillment_errors(&errors, None, false);
                error = true;
            }
        });
        // now we can take the return type of the given main function
        expected_return_type = main_fnsig.output();
    } else {
        // standard () main return type
        expected_return_type = ty::Binder::dummy(tcx.mk_unit());
    }

    if error {
        return;
    }

    let se_ty = tcx.mk_fn_ptr(expected_return_type.map_bound(|expected_return_type| {
        tcx.mk_fn_sig(iter::empty(), expected_return_type, false, hir::Unsafety::Normal, Abi::Rust)
    }));

    require_same_types(
        tcx,
        &ObligationCause::new(
            main_span,
            main_diagnostics_hir_id,
            ObligationCauseCode::MainFunctionType,
        ),
        se_ty,
        tcx.mk_fn_ptr(main_fnsig),
    );
}
fn check_start_fn_ty(tcx: TyCtxt<'_>, start_def_id: DefId) {
    let start_def_id = start_def_id.expect_local();
    let start_id = tcx.hir().local_def_id_to_hir_id(start_def_id);
    let start_span = tcx.def_span(start_def_id);
    let start_t = tcx.type_of(start_def_id);
    match start_t.kind() {
        ty::FnDef(..) => {
            if let Some(Node::Item(it)) = tcx.hir().find(start_id) {
                if let hir::ItemKind::Fn(ref sig, ref generics, _) = it.kind {
                    let mut error = false;
                    if !generics.params.is_empty() {
                        struct_span_err!(
                            tcx.sess,
                            generics.span,
                            E0132,
                            "start function is not allowed to have type parameters"
                        )
                        .span_label(generics.span, "start function cannot have type parameters")
                        .emit();
                        error = true;
                    }
                    if let Some(sp) = generics.where_clause.span() {
                        struct_span_err!(
                            tcx.sess,
                            sp,
                            E0647,
                            "start function is not allowed to have a `where` clause"
                        )
                        .span_label(sp, "start function cannot have a `where` clause")
                        .emit();
                        error = true;
                    }
                    if let hir::IsAsync::Async = sig.header.asyncness {
                        let span = tcx.sess.source_map().guess_head_span(it.span);
                        struct_span_err!(
                            tcx.sess,
                            span,
                            E0752,
                            "`start` is not allowed to be `async`"
                        )
                        .span_label(span, "`start` is not allowed to be `async`")
                        .emit();
                        error = true;
                    }

                    let attrs = tcx.hir().attrs(start_id);
                    for attr in attrs {
                        if attr.has_name(sym::track_caller) {
                            tcx.sess
                                .struct_span_err(
                                    attr.span,
                                    "`start` is not allowed to be `#[track_caller]`",
                                )
                                .span_label(
                                    start_span,
                                    "`start` is not allowed to be `#[track_caller]`",
                                )
                                .emit();
                            error = true;
                        }
                    }

                    if error {
                        return;
                    }
                }
            }

            let se_ty = tcx.mk_fn_ptr(ty::Binder::dummy(tcx.mk_fn_sig(
                [tcx.types.isize, tcx.mk_imm_ptr(tcx.mk_imm_ptr(tcx.types.u8))].iter().cloned(),
                tcx.types.isize,
                false,
                hir::Unsafety::Normal,
                Abi::Rust,
            )));

            require_same_types(
                tcx,
                &ObligationCause::new(start_span, start_id, ObligationCauseCode::StartFunctionType),
                se_ty,
                tcx.mk_fn_ptr(tcx.fn_sig(start_def_id)),
            );
        }
        _ => {
            span_bug!(start_span, "start has a non-function type: found `{}`", start_t);
        }
    }
}

fn check_for_entry_fn(tcx: TyCtxt<'_>) {
    match tcx.entry_fn(()) {
        Some((def_id, EntryFnType::Main)) => check_main_fn_ty(tcx, def_id),
        Some((def_id, EntryFnType::Start)) => check_start_fn_ty(tcx, def_id),
        _ => {}
    }
}

pub fn provide(providers: &mut Providers) {
    collect::provide(providers);
    coherence::provide(providers);
    check::provide(providers);
    variance::provide(providers);
    outlives::provide(providers);
    impl_wf_check::provide(providers);
    hir_wf_check::provide(providers);
}

pub fn check_crate(tcx: TyCtxt<'_>) -> Result<(), ErrorReported> {
    let _prof_timer = tcx.sess.timer("type_check_crate");

    // this ensures that later parts of type checking can assume that items
    // have valid types and not error
    // FIXME(matthewjasper) We shouldn't need to use `track_errors`.
    tcx.sess.track_errors(|| {
        tcx.sess.time("type_collecting", || {
            tcx.hir().for_each_module(|module| tcx.ensure().collect_mod_item_types(module))
        });
    })?;

    if tcx.features().rustc_attrs {
        tcx.sess.track_errors(|| {
            tcx.sess.time("outlives_testing", || outlives::test::test_inferred_outlives(tcx));
        })?;
    }

    tcx.sess.track_errors(|| {
        tcx.sess.time("impl_wf_inference", || impl_wf_check::impl_wf_check(tcx));
    })?;

    tcx.sess.track_errors(|| {
        tcx.sess.time("coherence_checking", || coherence::check_coherence(tcx));
    })?;

    if tcx.features().rustc_attrs {
        tcx.sess.track_errors(|| {
            tcx.sess.time("variance_testing", || variance::test::test_variance(tcx));
        })?;
    }

    tcx.sess.track_errors(|| {
        tcx.sess.time("wf_checking", || check::check_wf_new(tcx));
    })?;

    // NOTE: This is copy/pasted in librustdoc/core.rs and should be kept in sync.
    tcx.sess.time("item_types_checking", || {
        tcx.hir().for_each_module(|module| tcx.ensure().check_mod_item_types(module))
    });

    tcx.sess.time("item_bodies_checking", || tcx.typeck_item_bodies(()));

    check_unused::check_crate(tcx);
    check_for_entry_fn(tcx);

    if tcx.sess.err_count() == 0 { Ok(()) } else { Err(ErrorReported) }
}

/// A quasi-deprecated helper used in rustdoc and clippy to get
/// the type from a HIR node.
pub fn hir_ty_to_ty<'tcx>(tcx: TyCtxt<'tcx>, hir_ty: &hir::Ty<'_>) -> Ty<'tcx> {
    // In case there are any projections, etc., find the "environment"
    // def-ID that will be used to determine the traits/predicates in
    // scope.  This is derived from the enclosing item-like thing.
    let env_def_id = tcx.hir().get_parent_item(hir_ty.hir_id);
    let item_cx = self::collect::ItemCtxt::new(tcx, env_def_id.to_def_id());
    <dyn AstConv<'_>>::ast_ty_to_ty(&item_cx, hir_ty)
}

pub fn hir_trait_to_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    hir_trait: &hir::TraitRef<'_>,
    self_ty: Ty<'tcx>,
) -> Bounds<'tcx> {
    // In case there are any projections, etc., find the "environment"
    // def-ID that will be used to determine the traits/predicates in
    // scope.  This is derived from the enclosing item-like thing.
    let env_def_id = tcx.hir().get_parent_item(hir_trait.hir_ref_id);
    let item_cx = self::collect::ItemCtxt::new(tcx, env_def_id.to_def_id());
    let mut bounds = Bounds::default();
    let _ = <dyn AstConv<'_>>::instantiate_poly_trait_ref(
        &item_cx,
        hir_trait,
        DUMMY_SP,
        ty::BoundConstness::NotConst,
        self_ty,
        &mut bounds,
        true,
    );

    bounds
}
