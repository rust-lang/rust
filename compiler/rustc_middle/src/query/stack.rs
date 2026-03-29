use std::panic::Location;

use rustc_span::Span;

use crate::queries::TaggedQueryKey;

#[derive(Clone, Copy, Debug)]
pub struct QueryCallContext {
    pub span: Span,
    pub location: Option<&'static Location<'static>>,
}

/// Description of a frame in the query stack.
///
/// This is mostly used in case of cycles for error reporting.
#[derive(Debug)]
pub struct QueryStackFrame<'tcx> {
    pub call_context: QueryCallContext,

    /// The query and key of the query method call that this stack frame
    /// corresponds to.
    ///
    /// Code that doesn't care about the specific key can still use this to
    /// check which query it's for, or obtain the query's name.
    pub tagged_key: TaggedQueryKey<'tcx>,
}
