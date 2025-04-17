//! Construction of MIR from HIR.

// tidy-alphabetical-start
#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(try_blocks)]
// tidy-alphabetical-end

// The `builder` module used to be named `build`, but that was causing GitHub's
// "Go to file" feature to silently ignore all files in the module, probably
// because it assumes that "build" is a build-output directory. See #134365.
pub mod builder;
mod check_tail_calls;
mod check_unsafety;
mod errors;
pub mod thir;

use rustc_middle::util::Providers;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

pub fn provide(providers: &mut Providers) {
    providers.check_match = thir::pattern::check_match;
    providers.lit_to_const = thir::constant::lit_to_const;
    providers.closure_saved_names_of_captured_variables =
        builder::closure_saved_names_of_captured_variables;
    providers.check_unsafety = check_unsafety::check_unsafety;
    providers.check_tail_calls = check_tail_calls::check_tail_calls;
    providers.thir_body = thir::cx::thir_body;
}
