#![feature(associated_type_defaults)]
#![feature(box_patterns)]
#![feature(exact_size_is_empty)]
#![feature(let_chains)]
#![cfg_attr(bootstrap, feature(min_specialization))]
#![feature(try_blocks)]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

use rustc_middle::ty;

// Please change the public `use` directives cautiously, as they might be used by external tools.
// See issue #120130.
pub use self::drop_flag_effects::{
    drop_flag_effects_for_function_entry, drop_flag_effects_for_location,
    move_path_children_matching, on_all_children_bits, on_lookup_result_bits,
};
pub use self::framework::{
    fmt, graphviz, lattice, visit_results, Analysis, AnalysisDomain, Backward, Direction, Engine,
    Forward, GenKill, GenKillAnalysis, JoinSemiLattice, MaybeReachable, Results, ResultsCursor,
    ResultsVisitable, ResultsVisitor, SwitchIntEdgeEffects,
};
use self::move_paths::MoveData;

pub mod debuginfo;
pub mod drop_flag_effects;
pub mod elaborate_drops;
mod errors;
mod framework;
pub mod impls;
pub mod move_paths;
pub mod points;
pub mod rustc_peek;
pub mod storage;
pub mod un_derefer;
pub mod value_analysis;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

pub struct MoveDataParamEnv<'tcx> {
    pub move_data: MoveData<'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
}
