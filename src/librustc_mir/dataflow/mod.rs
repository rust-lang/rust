use rustc_ast::ast::{self, MetaItem};
use rustc_middle::ty;
use rustc_session::Session;
use rustc_span::symbol::{sym, Symbol};

pub(crate) use self::drop_flag_effects::*;
pub use self::framework::{
    visit_results, Analysis, AnalysisDomain, Backward, BorrowckFlowState, BorrowckResults,
    BottomValue, Engine, Forward, GenKill, GenKillAnalysis, Results, ResultsCursor,
    ResultsRefCursor, ResultsVisitor,
};

use self::move_paths::MoveData;

pub mod drop_flag_effects;
mod framework;
pub mod impls;
pub mod move_paths;

pub(crate) mod indexes {
    pub(crate) use super::{
        impls::borrows::BorrowIndex,
        move_paths::{InitIndex, MoveOutIndex, MovePathIndex},
    };
}

pub struct MoveDataParamEnv<'tcx> {
    pub(crate) move_data: MoveData<'tcx>,
    pub(crate) param_env: ty::ParamEnv<'tcx>,
}

pub(crate) fn has_rustc_mir_with(
    sess: &Session,
    attrs: &[ast::Attribute],
    name: Symbol,
) -> Option<MetaItem> {
    for attr in attrs {
        if sess.check_name(attr, sym::rustc_mir) {
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
