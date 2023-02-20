#![feature(associated_type_defaults)]
#![feature(box_patterns)]
#![feature(exact_size_is_empty)]
#![feature(let_chains)]
#![feature(min_specialization)]
#![feature(stmt_expr_attributes)]
#![feature(trusted_step)]
#![recursion_limit = "256"]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

use rustc_ast::MetaItem;
use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_fluent_macro::fluent_messages;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::symbol::{sym, Symbol};

pub use self::drop_flag_effects::{
    drop_flag_effects_for_function_entry, drop_flag_effects_for_location,
    move_path_children_matching, on_all_children_bits, on_all_drop_children_bits,
    on_lookup_result_bits,
};
pub use self::framework::{
    fmt, graphviz, lattice, visit_results, Analysis, AnalysisDomain, Backward, CallReturnPlaces,
    CloneAnalysis, Direction, Engine, Forward, GenKill, GenKillAnalysis, JoinSemiLattice, Results,
    ResultsCloned, ResultsClonedCursor, ResultsCursor, ResultsRefCursor, ResultsVisitable,
    ResultsVisitor, SwitchIntEdgeEffects,
};

use self::move_paths::MoveData;

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

pub(crate) mod indexes {
    pub(crate) use super::move_paths::MovePathIndex;
}

pub struct MoveDataParamEnv<'tcx> {
    pub move_data: MoveData<'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
}

pub fn has_rustc_mir_with(tcx: TyCtxt<'_>, def_id: DefId, name: Symbol) -> Option<MetaItem> {
    for attr in tcx.get_attrs(def_id, sym::rustc_mir) {
        let items = attr.meta_item_list();
        for item in items.iter().flat_map(|l| l.iter()) {
            match item.meta_item() {
                Some(mi) if mi.has_name(name) => return Some(mi.clone()),
                _ => continue,
            }
        }
    }
    None
}
