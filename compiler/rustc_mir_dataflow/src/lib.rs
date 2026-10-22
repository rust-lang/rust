// tidy-alphabetical-start
#![feature(associated_type_defaults)]
#![feature(deref_patterns)]
#![feature(exact_size_is_empty)]
#![feature(file_buffered)]
// tidy-alphabetical-end

use rustc_middle::ty;
use rustc_middle::util::Providers;

// Please change the public `use` directives cautiously, as they might be used by external tools.
// See issue #120130.
pub use self::drop_flag_effects::{
    DropFlagState, drop_flag_effects_for_function_entry, drop_flag_effects_for_location,
    move_path_children_matching, on_all_children_bits, on_lookup_result_bits,
};
pub use self::framework::{
    Analysis, Backward, Direction, EntryStates, Forward, GenKill, JoinSemiLattice, MaybeReachable,
    Results, ResultsCursor, ResultsVisitor, SwitchTargetIndex, fmt, graphviz, lattice,
    visit_results,
};
use self::move_paths::MoveData;

pub mod debuginfo;
mod diagnostics;
mod drop_flag_effects;
mod framework;
pub mod impls;
pub mod move_paths;
pub mod points;
mod pretty;
pub mod rustc_peek;
mod un_derefer;
pub mod value_analysis;

pub struct MoveDataTypingEnv<'tcx> {
    pub move_data: MoveData<'tcx>,
    pub typing_env: ty::TypingEnv<'tcx>,
}

pub fn provide(providers: &mut Providers) {
    providers.hooks.mir_pretty_extra_data = pretty::mir_pretty_extra_data;
}
