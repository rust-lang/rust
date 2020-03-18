//! The `DepGraphSafe` trait

use crate::ty::TyCtxt;

pub use rustc_query_system::dep_graph::{AssertDepGraphSafe, DepGraphSafe};

/// The type context itself can be used to access all kinds of tracked
/// state, but those accesses should always generate read events.
impl<'tcx> DepGraphSafe for TyCtxt<'tcx> {}
