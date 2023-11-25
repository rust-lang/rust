#![feature(associated_type_defaults)]
#![feature(box_patterns)]
#![feature(exact_size_is_empty)]
#![feature(let_chains)]
#![feature(min_specialization)]
#![feature(stmt_expr_attributes)]
#![recursion_limit = "256"]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_fluent_macro::fluent_messages;
use rustc_middle::ty;

pub use self::drop_flag_effects::{
    drop_flag_effects_for_function_entry, drop_flag_effects_for_location,
    move_path_children_matching, on_all_children_bits, on_lookup_result_bits,
};
pub use self::framework::{
    fmt, lattice, visit_results, Analysis, AnalysisDomain, Direction, GenKill, GenKillAnalysis,
    JoinSemiLattice, MaybeReachable, Results, ResultsCursor, ResultsVisitable, ResultsVisitor,
};
use self::framework::{Backward, CloneAnalysis, ResultsClonedCursor, SwitchIntEdgeEffects};
use self::move_paths::MoveData;

pub mod debuginfo;
pub mod drop_flag_effects;
pub mod elaborate_drops;
mod errors;
mod framework;
pub mod impls;
pub mod move_paths;
pub mod rustc_peek;
pub mod storage;
pub mod un_derefer;
pub mod value_analysis;

fluent_messages! { "../messages.ftl" }

pub struct MoveDataParamEnv<'tcx> {
    pub move_data: MoveData<'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
}
