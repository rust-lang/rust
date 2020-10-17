pub mod aggregate;
pub mod borrowck_errors;
pub mod elaborate_drops;
pub mod patch;
pub mod storage;

mod alignment;
pub mod collect_writes;
mod find_self_call;
mod graphviz;
pub(crate) mod pretty;
pub(crate) mod spanview;

pub use self::aggregate::expand_aggregate;
pub use self::alignment::is_disaligned;
pub use self::find_self_call::find_self_call;
pub use self::graphviz::write_node_label as write_graphviz_node_label;
pub use self::graphviz::{graphviz_safe_def_name, write_mir_graphviz};
pub use self::pretty::{dump_enabled, dump_mir, write_mir_pretty, PassWhere};
