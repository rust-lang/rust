// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::diagnostic_outside_of_impl)]
#![doc(rust_logo)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(decl_macro)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(never_type)]
#![feature(rustdoc_internals)]
#![feature(slice_ptr_get)]
#![feature(strict_overflow_ops)]
#![feature(trait_alias)]
#![feature(try_blocks)]
#![feature(unqualified_local_imports)]
#![feature(yeet_expr)]
#![warn(unqualified_local_imports)]
// tidy-alphabetical-end

pub mod check_consts;
pub mod const_eval;
mod errors;
pub mod interpret;
pub mod util;

use std::sync::atomic::AtomicBool;

use rustc_middle::ty;
use rustc_middle::util::Providers;

pub use self::errors::ReportErrorExt;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

pub fn provide(providers: &mut Providers) {
    const_eval::provide(providers);
    providers.tag_for_variant = const_eval::tag_for_variant_provider;
    providers.eval_to_const_value_raw = const_eval::eval_to_const_value_raw_provider;
    providers.eval_to_allocation_raw = const_eval::eval_to_allocation_raw_provider;
    providers.eval_static_initializer = const_eval::eval_static_initializer_provider;
    providers.hooks.const_caller_location = util::caller_location::const_caller_location_provider;
    providers.eval_to_valtree = |tcx, ty::PseudoCanonicalInput { typing_env, value }| {
        const_eval::eval_to_valtree(tcx, typing_env, value)
    };
    providers.hooks.try_destructure_mir_constant_for_user_output =
        const_eval::try_destructure_mir_constant_for_user_output;
    providers.valtree_to_const_val =
        |tcx, cv| const_eval::valtree_to_const_value(tcx, ty::TypingEnv::fully_monomorphized(), cv);
    providers.check_validity_requirement = |tcx, (init_kind, param_env_and_ty)| {
        util::check_validity_requirement(tcx, init_kind, param_env_and_ty)
    };
    providers.hooks.validate_scalar_in_layout =
        |tcx, scalar, layout| util::validate_scalar_in_layout(tcx, scalar, layout);
}

/// `rustc_driver::main` installs a handler that will set this to `true` if
/// the compiler has been sent a request to shut down, such as by a Ctrl-C.
/// This static lives here because it is only read by the interpreter.
pub static CTRL_C_RECEIVED: AtomicBool = AtomicBool::new(false);
