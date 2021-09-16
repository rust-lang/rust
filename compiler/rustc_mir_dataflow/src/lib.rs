#![feature(associated_type_defaults)]
#![feature(bool_to_option)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(const_panic)]
#![feature(exact_size_is_empty)]
#![feature(in_band_lifetimes)]
#![feature(iter_zip)]
#![feature(min_specialization)]
#![feature(once_cell)]
#![feature(stmt_expr_attributes)]
#![feature(trusted_step)]
#![recursion_limit = "256"]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

use rustc_ast::{self as ast, MetaItem};
use rustc_middle::ty;
use rustc_session::Session;
use rustc_span::symbol::{sym, Symbol};

pub use self::drop_flag_effects::{
    drop_flag_effects_for_function_entry, drop_flag_effects_for_location,
    move_path_children_matching, on_all_children_bits, on_all_drop_children_bits,
    on_lookup_result_bits,
};
pub use self::framework::{
    fmt, graphviz, lattice, visit_results, Analysis, AnalysisDomain, Backward, Direction, Engine,
    Forward, GenKill, GenKillAnalysis, JoinSemiLattice, Results, ResultsCursor, ResultsRefCursor,
    ResultsVisitable, ResultsVisitor,
};

use self::move_paths::MoveData;

pub mod drop_flag_effects;
pub mod elaborate_drops;
mod framework;
pub mod impls;
pub mod move_paths;
pub mod rustc_peek;
pub mod storage;

pub(crate) mod indexes {
    pub(crate) use super::move_paths::MovePathIndex;
}

pub struct MoveDataParamEnv<'tcx> {
    pub move_data: MoveData<'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
}

pub fn has_rustc_mir_with(
    _sess: &Session,
    attrs: &[ast::Attribute],
    name: Symbol,
) -> Option<MetaItem> {
    for attr in attrs {
        if attr.has_name(sym::rustc_mir) {
            let items = attr.meta_item_list();
            for item in items.iter().flat_map(|l| l.iter()) {
                match item.meta_item() {
                    Some(mi) if mi.has_name(name) => return Some(mi.clone()),
                    _ => continue,
                }
            }
        }
    }
    None
}
