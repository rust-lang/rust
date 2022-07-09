#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

mod errors;
mod impl_restriction;

use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_fluent_macro::fluent_messages;
use rustc_middle::query::Providers;

fluent_messages! { "../messages.ftl" }

pub fn provide(providers: &mut Providers) {
    impl_restriction::provide(providers);
}
