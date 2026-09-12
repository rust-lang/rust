//! Basic structs that end up being used in attributes for one reason or another,
//! but are not used exclusively in or around attributes.
#![deny(unstable_features, reason = "ends up in dependencies of rust-analyzer")]

mod collapse_macro_debug_info;
mod crate_type;
mod limit;
mod native_lib_kind;
mod sanitizer_set;

pub use collapse_macro_debug_info::CollapseMacroDebuginfo;
pub use crate_type::CrateType;
pub use limit::Limit;
pub use native_lib_kind::NativeLibKind;
pub use sanitizer_set::SanitizerSet;

#[cfg(test)]
mod tests;
