/*!

# typeck.rs

The type checker is responsible for:

1. Determining the type of each expression.
2. Resolving methods and traits.
3. Guaranteeing that most type rules are met. ("Most?", you say, "why most?"
   Well, dear reader, read on)

The main entry point is `check_crate()`. Type checking operates in
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
   documentation for the `check` module.

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

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]

#![allow(non_camel_case_types)]

#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(crate_visibility_modifier)]
#![feature(exhaustive_patterns)]
#![feature(nll)]
#![feature(refcell_replace_swap)]
#![feature(rustc_diagnostic_macros)]
#![feature(slice_patterns)]
#![feature(slice_sort_by_cached_key)]
#![feature(never_type)]

#![recursion_limit="256"]

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;
extern crate syntax_pos;

extern crate arena;

#[macro_use] extern crate rustc;
extern crate rustc_data_structures;
extern crate rustc_errors as errors;
extern crate rustc_target;
extern crate smallvec;

// N.B., this module needs to be declared first so diagnostics are
// registered before they are used.
mod diagnostics;

mod astconv;
mod check;
mod check_unused;
mod coherence;
mod collect;
mod constrained_type_params;
mod structured_errors;
mod impl_wf_check;
mod namespace;
mod outlives;
mod variance;

use rustc_target::spec::abi::Abi;
use rustc::hir::{self, Node};
use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::infer::InferOk;
use rustc::lint;
use rustc::middle;
use rustc::session;
use rustc::session::CompileIncomplete;
use rustc::session::config::{EntryFnType, nightly_options};
use rustc::traits::{ObligationCause, ObligationCauseCode, TraitEngine, TraitEngineExt};
use rustc::ty::subst::Substs;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::query::Providers;
use rustc::util;
use rustc::util::profiling::ProfileCategory;
use syntax_pos::Span;
use util::common::time;

use std::iter;

pub struct TypeAndSubsts<'tcx> {
    substs: &'tcx Substs<'tcx>,
    ty: Ty<'tcx>,
}

fn check_type_alias_enum_variants_enabled<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                                          span: Span) {
    if !tcx.features().type_alias_enum_variants {
        let mut err = tcx.sess.struct_span_err(
            span,
            "enum variants on type aliases are experimental"
        );
        if nightly_options::is_nightly_build() {
            help!(&mut err,
                "add `#![feature(type_alias_enum_variants)]` to the \
                crate attributes to enable");
        }
        err.emit();
    }
}

fn require_c_abi_if_variadic(tcx: TyCtxt,
                             decl: &hir::FnDecl,
                             abi: Abi,
                             span: Span) {
    if decl.variadic && !(abi == Abi::C || abi == Abi::Cdecl) {
        let mut err = struct_span_err!(tcx.sess, span, E0045,
            "variadic function must have C or cdecl calling convention");
        err.span_label(span, "variadics require C or cdecl calling convention").emit();
    }
}

fn require_same_types<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                cause: &ObligationCause<'tcx>,
                                expected: Ty<'tcx>,
                                actual: Ty<'tcx>)
                                -> bool {
    tcx.infer_ctxt().enter(|ref infcx| {
        let param_env = ty::ParamEnv::empty();
        let mut fulfill_cx = TraitEngine::new(infcx.tcx);
        match infcx.at(&cause, param_env).eq(expected, actual) {
            Ok(InferOk { obligations, .. }) => {
                fulfill_cx.register_predicate_obligations(infcx, obligations);
            }
            Err(err) => {
                infcx.report_mismatched_types(cause, expected, actual, err).emit();
                return false;
            }
        }

        match fulfill_cx.select_all_or_error(infcx) {
            Ok(()) => true,
            Err(errors) => {
                infcx.report_fulfillment_errors(&errors, None, false);
                false
            }
        }
    })
}

fn check_main_fn_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, main_def_id: DefId) {
    let main_id = tcx.hir().as_local_node_id(main_def_id).unwrap();
    let main_span = tcx.def_span(main_def_id);
    let main_t = tcx.type_of(main_def_id);
    match main_t.sty {
        ty::FnDef(..) => {
            if let Some(Node::Item(it)) = tcx.hir().find(main_id) {
                if let hir::ItemKind::Fn(.., ref generics, _) = it.node {
                    let mut error = false;
                    if !generics.params.is_empty() {
                        let msg = "`main` function is not allowed to have generic \
                                   parameters".to_owned();
                        let label = "`main` cannot have generic parameters".to_string();
                        struct_span_err!(tcx.sess, generics.span, E0131, "{}", msg)
                            .span_label(generics.span, label)
                            .emit();
                        error = true;
                    }
                    if let Some(sp) = generics.where_clause.span() {
                        struct_span_err!(tcx.sess, sp, E0646,
                            "`main` function is not allowed to have a `where` clause")
                            .span_label(sp, "`main` cannot have a `where` clause")
                            .emit();
                        error = true;
                    }
                    if error {
                        return;
                    }
                }
            }

            let actual = tcx.fn_sig(main_def_id);
            let expected_return_type = if tcx.lang_items().termination().is_some() {
                // we take the return type of the given main function, the real check is done
                // in `check_fn`
                actual.output().skip_binder()
            } else {
                // standard () main return type
                tcx.mk_unit()
            };

            let se_ty = tcx.mk_fn_ptr(ty::Binder::bind(
                tcx.mk_fn_sig(
                    iter::empty(),
                    expected_return_type,
                    false,
                    hir::Unsafety::Normal,
                    Abi::Rust
                )
            ));

            require_same_types(
                tcx,
                &ObligationCause::new(main_span, main_id, ObligationCauseCode::MainFunctionType),
                se_ty,
                tcx.mk_fn_ptr(actual));
        }
        _ => {
            span_bug!(main_span,
                      "main has a non-function type: found `{}`",
                      main_t);
        }
    }
}

fn check_start_fn_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, start_def_id: DefId) {
    let start_id = tcx.hir().as_local_node_id(start_def_id).unwrap();
    let start_span = tcx.def_span(start_def_id);
    let start_t = tcx.type_of(start_def_id);
    match start_t.sty {
        ty::FnDef(..) => {
            if let Some(Node::Item(it)) = tcx.hir().find(start_id) {
                if let hir::ItemKind::Fn(.., ref generics, _) = it.node {
                    let mut error = false;
                    if !generics.params.is_empty() {
                        struct_span_err!(tcx.sess, generics.span, E0132,
                            "start function is not allowed to have type parameters")
                            .span_label(generics.span,
                                        "start function cannot have type parameters")
                            .emit();
                        error = true;
                    }
                    if let Some(sp) = generics.where_clause.span() {
                        struct_span_err!(tcx.sess, sp, E0647,
                            "start function is not allowed to have a `where` clause")
                            .span_label(sp, "start function cannot have a `where` clause")
                            .emit();
                        error = true;
                    }
                    if error {
                        return;
                    }
                }
            }

            let se_ty = tcx.mk_fn_ptr(ty::Binder::bind(
                tcx.mk_fn_sig(
                    [
                        tcx.types.isize,
                        tcx.mk_imm_ptr(tcx.mk_imm_ptr(tcx.types.u8))
                    ].iter().cloned(),
                    tcx.types.isize,
                    false,
                    hir::Unsafety::Normal,
                    Abi::Rust
                )
            ));

            require_same_types(
                tcx,
                &ObligationCause::new(start_span, start_id, ObligationCauseCode::StartFunctionType),
                se_ty,
                tcx.mk_fn_ptr(tcx.fn_sig(start_def_id)));
        }
        _ => {
            span_bug!(start_span,
                      "start has a non-function type: found `{}`",
                      start_t);
        }
    }
}

fn check_for_entry_fn<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    match tcx.entry_fn(LOCAL_CRATE) {
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
}

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>)
                             -> Result<(), CompileIncomplete>
{
    tcx.sess.profiler(|p| p.start_activity(ProfileCategory::TypeChecking));

    // this ensures that later parts of type checking can assume that items
    // have valid types and not error
    tcx.sess.track_errors(|| {
        time(tcx.sess, "type collecting", ||
             collect::collect_item_types(tcx));

    })?;

    tcx.sess.track_errors(|| {
        time(tcx.sess, "outlives testing", ||
            outlives::test::test_inferred_outlives(tcx));
    })?;

    tcx.sess.track_errors(|| {
        time(tcx.sess, "impl wf inference", ||
             impl_wf_check::impl_wf_check(tcx));
    })?;

    tcx.sess.track_errors(|| {
      time(tcx.sess, "coherence checking", ||
          coherence::check_coherence(tcx));
    })?;

    tcx.sess.track_errors(|| {
        time(tcx.sess, "variance testing", ||
             variance::test::test_variance(tcx));
    })?;

    time(tcx.sess, "wf checking", || check::check_wf_new(tcx))?;

    time(tcx.sess, "item-types checking", || check::check_item_types(tcx))?;

    time(tcx.sess, "item-bodies checking", || check::check_item_bodies(tcx))?;

    check_unused::check_crate(tcx);
    check_for_entry_fn(tcx);

    tcx.sess.profiler(|p| p.end_activity(ProfileCategory::TypeChecking));

    tcx.sess.compile_status()
}

/// A quasi-deprecated helper used in rustdoc and save-analysis to get
/// the type from a HIR node.
pub fn hir_ty_to_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, hir_ty: &hir::Ty) -> Ty<'tcx> {
    // In case there are any projections etc, find the "environment"
    // def-id that will be used to determine the traits/predicates in
    // scope.  This is derived from the enclosing item-like thing.
    let env_node_id = tcx.hir().get_parent(hir_ty.id);
    let env_def_id = tcx.hir().local_def_id(env_node_id);
    let item_cx = self::collect::ItemCtxt::new(tcx, env_def_id);

    astconv::AstConv::ast_ty_to_ty(&item_cx, hir_ty)
}

pub fn hir_trait_to_predicates<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, hir_trait: &hir::TraitRef)
        -> (ty::PolyTraitRef<'tcx>, Vec<(ty::PolyProjectionPredicate<'tcx>, Span)>) {
    // In case there are any projections etc, find the "environment"
    // def-id that will be used to determine the traits/predicates in
    // scope.  This is derived from the enclosing item-like thing.
    let env_node_id = tcx.hir().get_parent(hir_trait.ref_id);
    let env_def_id = tcx.hir().local_def_id(env_node_id);
    let item_cx = self::collect::ItemCtxt::new(tcx, env_def_id);
    let mut projections = Vec::new();
    let (principal, _) = astconv::AstConv::instantiate_poly_trait_ref_inner(
        &item_cx, hir_trait, tcx.types.err, &mut projections, true
    );

    (principal, projections)
}

__build_diagnostic_array! { librustc_typeck, DIAGNOSTICS }
