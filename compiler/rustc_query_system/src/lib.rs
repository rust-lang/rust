// tidy-alphabetical-start
#![allow(internal_features)]
#![feature(assert_matches)]
#![feature(core_intrinsics)]
#![feature(hash_raw_entry)]
#![feature(let_chains)]
#![feature(min_specialization)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

pub mod cache;
pub mod dep_graph;
mod error;
pub mod ich;
pub mod query;
mod values;

pub use error::{HandleCycleError, LayoutOfDepth, QueryOverflow};
pub use values::Value;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
