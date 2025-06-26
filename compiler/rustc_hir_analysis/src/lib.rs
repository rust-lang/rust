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
#![feature(debug_closure_helpers)]
#![feature(gen_blocks)]
#![feature(if_let_guard)]
#![feature(iter_from_coroutine)]
#![feature(iter_intersperse)]
#![feature(never_type)]
#![feature(rustdoc_internals)]
#![feature(slice_partition_dedup)]
#![feature(try_blocks)]
#![feature(unwrap_infallible)]
// tidy-alphabetical-end

// These are used by Clippy.
pub mod check;

pub mod autoderef;
mod check_unused;
mod coherence;
mod collect;
mod constrained_generic_params;
mod delegation;
mod errors;
pub mod hir_ty_lowering;
pub mod hir_wf_check;
mod impl_wf_check;
mod outlives;
mod variance;

pub use errors::NoVariantNamed;
use rustc_abi::ExternAbi;
use rustc_hir::def::DefKind;
use rustc_hir::lints::DelayedLint;
use rustc_hir::{self as hir};
use rustc_middle::middle;
use rustc_middle::mir::interpret::GlobalId;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, Const, Ty, TyCtxt};
use rustc_session::parse::feature_err;
use rustc_span::symbol::sym;
use rustc_span::{ErrorGuaranteed, Span};
use rustc_trait_selection::traits;

pub use crate::collect::suggest_impl_trait;
use crate::hir_ty_lowering::{FeedConstTy, HirTyLowerer};

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

fn require_c_abi_if_c_variadic(
    tcx: TyCtxt<'_>,
    decl: &hir::FnDecl<'_>,
    abi: ExternAbi,
    span: Span,
) {
    // ABIs which can stably use varargs
    if !decl.c_variadic || matches!(abi, ExternAbi::C { .. } | ExternAbi::Cdecl { .. }) {
        return;
    }

    // ABIs with feature-gated stability
    let extended_abi_support = tcx.features().extended_varargs_abi_support();
    let extern_system_varargs = tcx.features().extern_system_varargs();

    // If the feature gate has been enabled, we can stop here
    if extern_system_varargs && let ExternAbi::System { .. } = abi {
        return;
    };
    if extended_abi_support && abi.supports_varargs() {
        return;
    };

    // Looks like we need to pick an error to emit.
    // Is there any feature which we could have enabled to make this work?
    let unstable_explain =
        format!("C-variadic functions with the {abi} calling convention are unstable");
    match abi {
        ExternAbi::System { .. } => {
            feature_err(&tcx.sess, sym::extern_system_varargs, span, unstable_explain)
        }
        abi if abi.supports_varargs() => {
            feature_err(&tcx.sess, sym::extended_varargs_abi_support, span, unstable_explain)
        }
        _ => tcx.dcx().create_err(errors::VariadicFunctionCompatibleConvention {
            span,
            convention: &format!("{abi}"),
        }),
    }
    .emit();
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
        enforce_impl_non_lifetime_params_are_constrained:
            impl_wf_check::enforce_impl_non_lifetime_params_are_constrained,
        ..*providers
    };
}

fn emit_delayed_lint(lint: &DelayedLint, tcx: TyCtxt<'_>) {
    match lint {
        DelayedLint::AttributeParsing(attribute_lint) => {
            rustc_attr_parsing::emit_attribute_lint(attribute_lint, tcx)
        }
    }
}

pub fn check_crate(tcx: TyCtxt<'_>) {
    let _prof_timer = tcx.sess.timer("type_check_crate");

    tcx.sess.time("coherence_checking", || {
        // When discarding query call results, use an explicit type to indicate
        // what we are intending to discard, to help future type-based refactoring.
        type R = Result<(), ErrorGuaranteed>;

        let _: R = tcx.ensure_ok().check_type_wf(());

        for &trait_def_id in tcx.all_local_trait_impls(()).keys() {
            let _: R = tcx.ensure_ok().coherent_trait(trait_def_id);
        }
        // these queries are executed for side-effects (error reporting):
        let _: R = tcx.ensure_ok().crate_inherent_impls_validity_check(());
        let _: R = tcx.ensure_ok().crate_inherent_impls_overlap_check(());
    });

    tcx.sess.time("emit_ast_lowering_delayed_lints", || {
        // sanity check in debug mode that all lints are really noticed
        // and we really will emit them all in the loop right below.
        //
        // during ast lowering, when creating items, foreign items, trait items and impl items
        // we store in them whether they have any lints in their owner node that should be
        // picked up by `hir_crate_items`. However, theoretically code can run between that
        // boolean being inserted into the item and the owner node being created.
        // We don't want any new lints to be emitted there
        // (though honestly, you have to really try to manage to do that but still),
        // but this check is there to catch that.
        #[cfg(debug_assertions)]
        {
            // iterate over all owners
            for owner_id in tcx.hir_crate_items(()).owners() {
                // if it has delayed lints
                if let Some(delayed_lints) = tcx.opt_ast_lowering_delayed_lints(owner_id) {
                    if !delayed_lints.lints.is_empty() {
                        // assert that delayed_lint_items also picked up this item to have lints
                        assert!(
                            tcx.hir_crate_items(()).delayed_lint_items().any(|i| i == owner_id)
                        );
                    }
                }
            }
        }

        for owner_id in tcx.hir_crate_items(()).delayed_lint_items() {
            if let Some(delayed_lints) = tcx.opt_ast_lowering_delayed_lints(owner_id) {
                for lint in &delayed_lints.lints {
                    emit_delayed_lint(lint, tcx);
                }
            }
        }
    });

    tcx.par_hir_body_owners(|item_def_id| {
        let def_kind = tcx.def_kind(item_def_id);
        // Make sure we evaluate all static and (non-associated) const items, even if unused.
        // If any of these fail to evaluate, we do not want this crate to pass compilation.
        match def_kind {
            DefKind::Static { .. } => {
                tcx.ensure_ok().eval_static_initializer(item_def_id);
                check::maybe_check_static_with_link_section(tcx, item_def_id);
            }
            DefKind::Const if !tcx.generics_of(item_def_id).own_requires_monomorphization() => {
                // FIXME(generic_const_items): Passing empty instead of identity args is fishy but
                //                             seems to be fine for now. Revisit this!
                let instance = ty::Instance::new_raw(item_def_id.into(), ty::GenericArgs::empty());
                let cid = GlobalId { instance, promoted: None };
                let typing_env = ty::TypingEnv::fully_monomorphized();
                tcx.ensure_ok().eval_to_const_value_raw(typing_env.as_query_input(cid));
            }
            _ => (),
        }
        // Skip `AnonConst`s because we feed their `type_of`.
        if !matches!(def_kind, DefKind::AnonConst) {
            tcx.ensure_ok().typeck(item_def_id);
        }
        // Ensure we generate the new `DefId` before finishing `check_crate`.
        // Afterwards we freeze the list of `DefId`s.
        if tcx.needs_coroutine_by_move_body_def_id(item_def_id.to_def_id()) {
            tcx.ensure_done().coroutine_by_move_body_def_id(item_def_id);
        }
    });

    if tcx.features().rustc_attrs() {
        tcx.sess.time("dumping_rustc_attr_data", || {
            outlives::dump::inferred_outlives(tcx);
            variance::dump::variances(tcx);
            collect::dump::opaque_hidden_types(tcx);
            collect::dump::predicates_and_item_bounds(tcx);
            collect::dump::def_parents(tcx);
            collect::dump::vtables(tcx);
        });
    }

    tcx.ensure_ok().check_unused_traits(());
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
    let env_def_id = tcx.hir_get_parent_item(hir_ty.hir_id);
    collect::ItemCtxt::new(tcx, env_def_id.def_id)
        .lowerer()
        .lower_ty_maybe_return_type_notation(hir_ty)
}

/// This is for rustdoc.
// FIXME(const_generics): having special methods for rustdoc in `rustc_hir_analysis` is cursed
pub fn lower_const_arg_for_rustdoc<'tcx>(
    tcx: TyCtxt<'tcx>,
    hir_ct: &hir::ConstArg<'tcx>,
    feed: FeedConstTy<'_, 'tcx>,
) -> Const<'tcx> {
    let env_def_id = tcx.hir_get_parent_item(hir_ct.hir_id);
    collect::ItemCtxt::new(tcx, env_def_id.def_id).lowerer().lower_const_arg(hir_ct, feed)
}
