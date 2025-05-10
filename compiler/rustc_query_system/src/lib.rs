// tidy-alphabetical-start
#![allow(rustc::potential_query_instability, internal_features)]
#![feature(assert_matches)]
#![feature(core_intrinsics)]
#![feature(dropck_eyepatch)]
#![feature(min_specialization)]
// tidy-alphabetical-end

pub mod cache;
pub mod dep_graph;
mod error;
pub mod ich;
pub mod query;
mod values;

pub use error::{HandleCycleError, QueryOverflow, QueryOverflowNote};
pub use values::Value;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
