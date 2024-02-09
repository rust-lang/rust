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
  all subtyping and assignment constraints are met. In essence, the check
  module specifies the constraints, and the infer module solves them.

## Note

This API is completely unstable and subject to change.

*/

#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::potential_query_instability)]
#![allow(rustc::untranslatable_diagnostic)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(rustdoc_internals)]
#![allow(internal_features)]
#![feature(control_flow_enum)]
#![feature(if_let_guard)]
#![feature(is_sorted)]
#![feature(iter_intersperse)]
#![feature(let_chains)]
#![cfg_attr(bootstrap, feature(min_specialization))]
#![feature(never_type)]
#![feature(lazy_cell)]
#![feature(slice_partition_dedup)]
#![feature(try_blocks)]

#[macro_use]
extern crate tracing;

#[macro_use]
extern crate rustc_middle;

// These are used by Clippy.
pub mod check;

pub mod astconv;
pub mod autoderef;
mod bounds;
mod check_unused;
mod coherence;
// FIXME: This module shouldn't be public.
pub mod collect;
mod constrained_generic_params;
mod errors;
pub mod hir_wf_check;
mod impl_wf_check;
mod outlives;
pub mod structured_errors;
mod variance;

use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_middle::middle;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::util;
use rustc_session::parse::feature_err;
use rustc_span::{symbol::sym, Span, DUMMY_SP};
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits;

use astconv::{AstConv, OnlySelfBounds};
use bounds::Bounds;
use rustc_hir::def::DefKind;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

fn require_c_abi_if_c_variadic(tcx: TyCtxt<'_>, decl: &hir::FnDecl<'_>, abi: Abi, span: Span) {
    const CONVENTIONS_UNSTABLE: &str =
        "`C`, `cdecl`, `system`, `aapcs`, `win64`, `sysv64` or `efiapi`";
    const CONVENTIONS_STABLE: &str = "`C` or `cdecl`";
    const UNSTABLE_EXPLAIN: &str =
        "using calling conventions other than `C` or `cdecl` for varargs functions is unstable";

    if !decl.c_variadic || matches!(abi, Abi::C { .. } | Abi::Cdecl { .. }) {
        return;
    }

    let extended_abi_support = tcx.features().extended_varargs_abi_support;
    let conventions = match (extended_abi_support, abi.supports_varargs()) {
        // User enabled additional ABI support for varargs and function ABI matches those ones.
        (true, true) => return,

        // Using this ABI would be ok, if the feature for additional ABI support was enabled.
        // Return CONVENTIONS_STABLE, because we want the other error to look the same.
        (false, true) => {
            feature_err(&tcx.sess, sym::extended_varargs_abi_support, span, UNSTABLE_EXPLAIN)
                .emit();
            CONVENTIONS_STABLE
        }

        (false, false) => CONVENTIONS_STABLE,
        (true, false) => CONVENTIONS_UNSTABLE,
    };

    tcx.dcx().emit_err(errors::VariadicFunctionCompatibleConvention { span, conventions });
}

pub fn provide(providers: &mut Providers) {
    collect::provide(providers);
    coherence::provide(providers);
    check::provide(providers);
    check_unused::provide(providers);
    variance::provide(providers);
    outlives::provide(providers);
    impl_wf_check::provide(providers);
    hir_wf_check::provide(providers);
}

pub fn check_crate(tcx: TyCtxt<'_>) -> Result<(), ErrorGuaranteed> {
    let _prof_timer = tcx.sess.timer("type_check_crate");

    // this ensures that later parts of type checking can assume that items
    // have valid types and not error
    tcx.sess.time("type_collecting", || {
        tcx.hir().for_each_module(|module| tcx.ensure().collect_mod_item_types(module))
    });

    if tcx.features().rustc_attrs {
        tcx.sess.time("outlives_testing", || outlives::test::test_inferred_outlives(tcx))?;
    }

    tcx.sess.time("coherence_checking", || {
        // Check impls constrain their parameters
        let res =
            tcx.hir().try_par_for_each_module(|module| tcx.ensure().check_mod_impl_wf(module));

        for &trait_def_id in tcx.all_local_trait_impls(()).keys() {
            let _ = tcx.ensure().coherent_trait(trait_def_id);
        }
        // these queries are executed for side-effects (error reporting):
        res.and(tcx.ensure().crate_inherent_impls(()))
            .and(tcx.ensure().crate_inherent_impls_overlap_check(()))
    })?;

    if tcx.features().rustc_attrs {
        tcx.sess.time("variance_testing", || variance::test::test_variance(tcx))?;
    }

    tcx.sess.time("wf_checking", || {
        tcx.hir().par_for_each_module(|module| {
            let _ = tcx.ensure().check_mod_type_wf(module);
        })
    });

    if tcx.features().rustc_attrs {
        collect::test_opaque_hidden_types(tcx)?;
    }

    // Freeze definitions as we don't add new ones at this point. This improves performance by
    // allowing lock-free access to them.
    tcx.untracked().definitions.freeze();

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

    Ok(())
}

/// A quasi-deprecated helper used in rustdoc and clippy to get
/// the type from a HIR node.
pub fn hir_ty_to_ty<'tcx>(tcx: TyCtxt<'tcx>, hir_ty: &hir::Ty<'tcx>) -> Ty<'tcx> {
    // In case there are any projections, etc., find the "environment"
    // def-ID that will be used to determine the traits/predicates in
    // scope. This is derived from the enclosing item-like thing.
    let env_def_id = tcx.hir().get_parent_item(hir_ty.hir_id);
    let item_cx = self::collect::ItemCtxt::new(tcx, env_def_id.def_id);
    item_cx.astconv().ast_ty_to_ty(hir_ty)
}

pub fn hir_trait_to_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    hir_trait: &hir::TraitRef<'tcx>,
    self_ty: Ty<'tcx>,
) -> Bounds<'tcx> {
    // In case there are any projections, etc., find the "environment"
    // def-ID that will be used to determine the traits/predicates in
    // scope. This is derived from the enclosing item-like thing.
    let env_def_id = tcx.hir().get_parent_item(hir_trait.hir_ref_id);
    let item_cx = self::collect::ItemCtxt::new(tcx, env_def_id.def_id);
    let mut bounds = Bounds::default();
    let _ = &item_cx.astconv().instantiate_poly_trait_ref(
        hir_trait,
        DUMMY_SP,
        ty::BoundConstness::NotConst,
        ty::ImplPolarity::Positive,
        self_ty,
        &mut bounds,
        true,
        OnlySelfBounds(false),
    );

    bounds
}
