use std::fmt::Debug;

use rustc_span::def_id::DefId;

use crate::dep_graph::DepKind;
use crate::queries::TaggedQueryKey;

/// Description of a frame in the query stack.
///
/// This is mostly used in case of cycles for error reporting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryStackFrame<'tcx> {
    pub tagged_key: TaggedQueryKey<'tcx>,
    pub dep_kind: DepKind,
    pub def_id: Option<DefId>,
}
