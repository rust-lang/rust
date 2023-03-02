#![feature(assert_matches)]
#![feature(core_intrinsics)]
#![feature(hash_raw_entry)]
#![feature(min_specialization)]
#![feature(extern_types)]
#![feature(let_chains)]
#![allow(rustc::potential_query_instability)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_data_structures;
#[macro_use]
extern crate rustc_macros;

use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_macros::fluent_messages;

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

fluent_messages! { "../messages.ftl" }
