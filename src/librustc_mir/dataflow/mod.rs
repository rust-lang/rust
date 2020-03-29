use rustc_ast::ast::{self, MetaItem};
use rustc_middle::ty;
use rustc_span::symbol::{sym, Symbol};

pub(crate) use self::drop_flag_effects::*;
pub use self::framework::{
    visit_results, Analysis, AnalysisDomain, BorrowckFlowState, BorrowckResults, BottomValue,
    Engine, GenKill, GenKillAnalysis, Results, ResultsCursor, ResultsRefCursor, ResultsVisitor,
};
pub use self::impls::{
    borrows::Borrows, DefinitelyInitializedPlaces, EverInitializedPlaces, MaybeBorrowedLocals,
    MaybeInitializedPlaces, MaybeMutBorrowedLocals, MaybeRequiresStorage, MaybeStorageLive,
    MaybeUninitializedPlaces,
};

use self::move_paths::MoveData;

pub mod drop_flag_effects;
mod framework;
mod impls;
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

pub(crate) fn has_rustc_mir_with(attrs: &[ast::Attribute], name: Symbol) -> Option<MetaItem> {
    for attr in attrs {
        if attr.check_name(sym::rustc_mir) {
            let items = attr.meta_item_list();
            for item in items.iter().flat_map(|l| l.iter()) {
                match item.meta_item() {
                    Some(mi) if mi.check_name(name) => return Some(mi.clone()),
                    _ => continue,
                }
            }
        }
    }
    None
}
