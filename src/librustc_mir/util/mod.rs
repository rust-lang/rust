use core::unicode::property::Pattern_White_Space;
use rustc::ty::TyCtxt;
use syntax_pos::Span;

pub mod aggregate;
pub mod borrowck_errors;
pub mod elaborate_drops;
pub mod def_use;
pub mod patch;

mod alignment;
mod graphviz;
pub(crate) mod pretty;
pub mod liveness;
pub mod collect_writes;

pub use self::aggregate::expand_aggregate;
pub use self::alignment::is_disaligned;
pub use self::pretty::{dump_enabled, dump_mir, write_mir_pretty, PassWhere};
pub use self::graphviz::{graphviz_safe_def_name, write_mir_graphviz};
pub use self::graphviz::write_node_label as write_graphviz_node_label;

/// If possible, suggest replacing `ref` with `ref mut`.
pub fn suggest_ref_mut(tcx: TyCtxt<'_>, binding_span: Span) -> Option<(String)> {
    let hi_src = tcx.sess.source_map().span_to_snippet(binding_span).unwrap();
    if hi_src.starts_with("ref")
        && hi_src["ref".len()..].starts_with(Pattern_White_Space)
    {
        let replacement = format!("ref mut{}", &hi_src["ref".len()..]);
        Some(replacement)
    } else {
        None
    }
}
