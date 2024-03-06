#![feature(assert_matches)]
#![feature(core_intrinsics)]
#![feature(generic_nonzero)]
#![feature(hash_raw_entry)]
#![feature(min_specialization)]
#![feature(let_chains)]
#![allow(rustc::potential_query_instability, internal_features)]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_data_structures;
#[macro_use]
extern crate rustc_macros;

pub mod cache;
pub mod dep_graph;
mod error;
pub mod ich;
pub mod query;
mod values;

pub use error::HandleCycleError;
pub use error::LayoutOfDepth;
pub use error::QueryOverflow;
pub use values::Value;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
