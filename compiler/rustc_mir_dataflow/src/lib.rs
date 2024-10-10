// tidy-alphabetical-start
#![feature(assert_matches)]
#![feature(associated_type_defaults)]
#![feature(box_patterns)]
#![feature(exact_size_is_empty)]
#![feature(file_buffered)]
#![feature(let_chains)]
#![feature(try_blocks)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

use rustc_middle::ty;

// Please change the public `use` directives cautiously, as they might be used by external tools.
// See issue #120130.
pub use self::drop_flag_effects::{
    drop_flag_effects_for_function_entry, drop_flag_effects_for_location,
    move_path_children_matching, on_all_children_bits, on_lookup_result_bits,
};
pub use self::framework::{
    Analysis, Backward, Direction, Engine, Forward, GenKill, JoinSemiLattice, MaybeReachable,
    Results, ResultsCursor, ResultsVisitable, ResultsVisitor, SwitchIntEdgeEffects, fmt, graphviz,
    lattice, visit_results,
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
