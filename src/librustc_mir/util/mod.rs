use core::unicode::property::Pattern_White_Space;
use rustc::ty;
use syntax_pos::Span;

pub mod borrowck_errors;
pub mod def_use;
pub mod elaborate_drops;
pub mod patch;

mod alignment;
pub mod collect_writes;
mod graphviz;
pub mod liveness;
pub(crate) mod pretty;

pub use self::alignment::is_disaligned;
pub use self::graphviz::write_mir_graphviz;
pub use self::graphviz::write_node_label as write_graphviz_node_label;
pub use self::pretty::{dump_enabled, dump_mir, write_mir_pretty, PassWhere};

/// If possible, suggest replacing `ref` with `ref mut`.
pub fn suggest_ref_mut<'cx, 'gcx, 'tcx>(
    tcx: ty::TyCtxt<'cx, 'gcx, 'tcx>,
    binding_span: Span,
) -> Option<(String)> {
    let hi_src = tcx.sess.source_map().span_to_snippet(binding_span).unwrap();
    if hi_src.starts_with("ref") && hi_src["ref".len()..].starts_with(Pattern_White_Space) {
        let replacement = format!("ref mut{}", &hi_src["ref".len()..]);
        Some(replacement)
    } else {
        None
    }
}
