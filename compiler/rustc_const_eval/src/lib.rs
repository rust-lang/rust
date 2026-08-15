// tidy-alphabetical-start
#![feature(array_try_map)]
#![feature(decl_macro)]
#![feature(deref_patterns)]
#![feature(never_type)]
#![feature(slice_ptr_get)]
#![feature(trait_alias)]
#![feature(unqualified_local_imports)]
#![feature(yeet_expr)]
#![warn(unqualified_local_imports)]
// tidy-alphabetical-end

pub mod check_consts;
pub mod const_eval;
mod diagnostics;
pub mod interpret;
pub mod util;

use std::sync::atomic::AtomicBool;

use rustc_middle::util::Providers;
use rustc_middle::{bug, ty};

/// Const eval always happens in post analysis mode in order to be able to use the hidden types of
/// opaque types. This is needed for trivial things like `size_of`, but also for using associated
/// types that are not specified in the opaque type. We also use MIR bodies whose opaque types have
/// already been revealed, so we'd be able to at least partially observe the hidden types anyways.
fn assert_typing_mode(typing_mode: ty::TypingMode<'_>) {
    if cfg!(debug_assertions) {
        match typing_mode.assert_not_erased() {
            ty::TypingMode::PostAnalysis | ty::TypingMode::Codegen => {}
            // Const eval always happens in PostAnalysis or Codegen mode. See the comment in
            // `InterpCx::new` for more details.
            ty::TypingMode::Coherence
            | ty::TypingMode::Typeck { .. }
            | ty::TypingMode::Reflection
            | ty::TypingMode::PostTypeckUntilBorrowck { .. }
            | ty::TypingMode::PostBorrowck { .. } => bug!(
                "Const eval should always happens in PostAnalysis or Codegen mode. See the comment on `assert_typing_mode` for more details."
            ),
        }
    }
}

pub fn provide(providers: &mut Providers) {
    const_eval::provide(&mut providers.queries);
    providers.queries.tag_for_variant = const_eval::tag_for_variant_provider;
    providers.queries.eval_to_const_value_raw = const_eval::eval_to_const_value_raw_provider;
    providers.queries.eval_to_allocation_raw = const_eval::eval_to_allocation_raw_provider;
    providers.queries.eval_static_initializer = const_eval::eval_static_initializer_provider;
    providers.hooks.const_caller_location = util::caller_location::const_caller_location_provider;
    providers.queries.eval_to_valtree = |tcx, ty::PseudoCanonicalInput { typing_env, value }| {
        const_eval::eval_to_valtree(tcx, typing_env, value)
    };
    providers.hooks.try_destructure_mir_constant_for_user_output =
        const_eval::try_destructure_mir_constant_for_user_output;
    providers.queries.valtree_to_const_val =
        |tcx, cv| const_eval::valtree_to_const_value(tcx, ty::TypingEnv::fully_monomorphized(), cv);
    providers.queries.check_validity_requirement = |tcx, (init_kind, param_env_and_ty)| {
        util::check_validity_requirement(tcx, init_kind, param_env_and_ty)
    };
    providers.hooks.validate_scalar_in_layout =
        |tcx, scalar, layout| util::validate_scalar_in_layout(tcx, scalar, layout);
}

/// `rustc_driver::main` installs a handler that will set this to `true` if
/// the compiler has been sent a request to shut down, such as by a Ctrl-C.
/// This static lives here because it is only read by the interpreter.
pub static CTRL_C_RECEIVED: AtomicBool = AtomicBool::new(false);
