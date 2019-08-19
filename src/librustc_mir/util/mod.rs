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
