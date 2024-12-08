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
   containing function). Inference is used to supply types wherever
   they are unknown. The actual checking of a function itself has
   several phases (check, regionck, writeback), as discussed in the
   documentation for the [`check`] module.

The type checker is defined into various submodules which are documented
independently:

- hir_ty_lowering: lowers type-system entities from the [HIR][hir] to the
  [`rustc_middle::ty`] representation.

- collect: computes the types of each top-level item and enters them into
  the `tcx.types` table for later use.

- coherence: enforces coherence rules, builds some tables.

- variance: variance inference

- outlives: outlives inference

- check: walks over function bodies and type checks them, inferring types for
  local variables, type parameters, etc as necessary.

- infer: finds the types to use for each type variable such that
  all subtyping and assignment constraints are met. In essence, the check
  module specifies the constraints, and the infer module solves them.

## Note

This API is completely unstable and subject to change.

*/

// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(assert_matches)]
#![feature(coroutines)]
#![feature(if_let_guard)]
#![feature(iter_from_coroutine)]
#![feature(iter_intersperse)]
#![feature(let_chains)]
#![feature(never_type)]
#![feature(rustdoc_internals)]
#![feature(slice_partition_dedup)]
#![feature(try_blocks)]
#![feature(unwrap_infallible)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

// These are used by Clippy.
pub mod check;

pub mod autoderef;
mod bounds;
mod check_unused;
mod coherence;
mod delegation;
pub mod hir_ty_lowering;
// FIXME: This module shouldn't be public.
pub mod collect;
mod constrained_generic_params;
mod errors;
pub mod hir_wf_check;
mod impl_wf_check;
mod outlives;
mod variance;

use rustc_abi::ExternAbi;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_middle::middle;
use rustc_middle::mir::interpret::GlobalId;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, Const, Ty, TyCtxt};
use rustc_span::Span;
use rustc_trait_selection::traits;

use self::hir_ty_lowering::{FeedConstTy, HirTyLowerer};

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

fn require_c_abi_if_c_variadic(
    tcx: TyCtxt<'_>,
    decl: &hir::FnDecl<'_>,
    abi: ExternAbi,
    span: Span,
) {
    if decl.c_variadic && !abi.supports_varargs() {
        tcx.dcx().emit_err(errors::VariadicFunctionCompatibleConvention { span });
    }
}

pub fn provide(providers: &mut Providers) {
    collect::provide(providers);
    coherence::provide(providers);
    check::provide(providers);
    check_unused::provide(providers);
    variance::provide(providers);
    outlives::provide(providers);
    hir_wf_check::provide(providers);
    *providers = Providers {
        inherit_sig_for_delegation_item: delegation::inherit_sig_for_delegation_item,
        ..*providers
    };
}

pub fn check_crate(tcx: TyCtxt<'_>) {
    let _prof_timer = tcx.sess.timer("type_check_crate");

    tcx.sess.time("coherence_checking", || {
        tcx.hir().par_for_each_module(|module| {
            let _ = tcx.ensure().check_mod_type_wf(module);
        });

        for &trait_def_id in tcx.all_local_trait_impls(()).keys() {
            let _ = tcx.ensure().coherent_trait(trait_def_id);
        }
        // these queries are executed for side-effects (error reporting):
        let _ = tcx.ensure().crate_inherent_impls_validity_check(());
        let _ = tcx.ensure().crate_inherent_impls_overlap_check(());
    });

    if tcx.features().rustc_attrs() {
        tcx.sess.time("outlives_dumping", || outlives::dump::inferred_outlives(tcx));
        tcx.sess.time("variance_dumping", || variance::dump::variances(tcx));
        collect::dump::opaque_hidden_types(tcx);
        collect::dump::predicates_and_item_bounds(tcx);
        collect::dump::def_parents(tcx);
    }

    // Make sure we evaluate all static and (non-associated) const items, even if unused.
    // If any of these fail to evaluate, we do not want this crate to pass compilation.
    tcx.hir().par_body_owners(|item_def_id| {
        let def_kind = tcx.def_kind(item_def_id);
        match def_kind {
            DefKind::Static { .. } => tcx.ensure().eval_static_initializer(item_def_id),
            DefKind::Const if tcx.generics_of(item_def_id).is_empty() => {
                let instance = ty::Instance::new(item_def_id.into(), ty::GenericArgs::empty());
                let cid = GlobalId { instance, promoted: None };
                let typing_env = ty::TypingEnv::fully_monomorphized();
                tcx.ensure().eval_to_const_value_raw(typing_env.as_query_input(cid));
            }
            _ => (),
        }
    });

    // FIXME: Remove this when we implement creating `DefId`s
    // for anon constants during their parents' typeck.
    // Typeck all body owners in parallel will produce queries
    // cycle errors because it may typeck on anon constants directly.
    tcx.hir().par_body_owners(|item_def_id| {
        let def_kind = tcx.def_kind(item_def_id);
        if !matches!(def_kind, DefKind::AnonConst) {
            tcx.ensure().typeck(item_def_id);
        }
    });

    tcx.ensure().check_unused_traits(());
}

/// Lower a [`hir::Ty`] to a [`Ty`].
///
/// <div class="warning">
///
/// This function is **quasi-deprecated**. It can cause ICEs if called inside of a body
/// (of a function or constant) and especially if it contains inferred types (`_`).
///
/// It's used in rustdoc and Clippy.
///
/// </div>
pub fn lower_ty<'tcx>(tcx: TyCtxt<'tcx>, hir_ty: &hir::Ty<'tcx>) -> Ty<'tcx> {
    // In case there are any projections, etc., find the "environment"
    // def-ID that will be used to determine the traits/predicates in
    // scope. This is derived from the enclosing item-like thing.
    let env_def_id = tcx.hir().get_parent_item(hir_ty.hir_id);
    collect::ItemCtxt::new(tcx, env_def_id.def_id).lower_ty(hir_ty)
}

/// This is for rustdoc.
// FIXME(const_generics): having special methods for rustdoc in `rustc_hir_analysis` is cursed
pub fn lower_const_arg_for_rustdoc<'tcx>(
    tcx: TyCtxt<'tcx>,
    hir_ct: &hir::ConstArg<'tcx>,
    feed: FeedConstTy,
) -> Const<'tcx> {
    let env_def_id = tcx.hir().get_parent_item(hir_ct.hir_id);
    collect::ItemCtxt::new(tcx, env_def_id.def_id).lowerer().lower_const_arg(hir_ct, feed)
}
