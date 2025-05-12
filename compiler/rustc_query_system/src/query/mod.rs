mod plumbing;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::transmute;
use std::sync::Arc;

pub use self::plumbing::*;

mod job;
pub use self::job::{
    QueryInfo, QueryJob, QueryJobId, QueryJobInfo, QueryMap, break_query_cycles, print_query_stack,
    report_cycle,
};

mod caches;
pub use self::caches::{DefIdCache, DefaultCache, QueryCache, SingleCache, VecCache};

mod config;
use rustc_data_structures::jobserver::Proxy;
use rustc_data_structures::sync::{DynSend, DynSync};
use rustc_errors::DiagInner;
use rustc_hashes::Hash64;
use rustc_hir::def::DefKind;
use rustc_macros::{Decodable, Encodable};
use rustc_span::Span;
use rustc_span::def_id::DefId;

pub use self::config::{HashResult, QueryConfig};
use crate::dep_graph::{DepKind, DepNodeIndex, HasDepContext, SerializedDepNodeIndex};

/// Description of a frame in the query stack.
///
/// This is mostly used in case of cycles for error reporting.
#[derive(Clone, Debug)]
pub struct QueryStackFrame<I> {
    /// This field initially stores a `QueryStackDeferred` during collection,
    /// but can later be changed to `QueryStackFrameExtra` containing concrete information
    /// by calling `lift`. This is done so that collecting query does not need to invoke
    /// queries, instead `lift` will call queries in a more appropriate location.
    pub info: I,

    pub dep_kind: DepKind,
    /// This hash is used to deterministically pick
    /// a query to remove cycles in the parallel compiler.
    hash: Hash64,
    pub def_id: Option<DefId>,
    /// A def-id that is extracted from a `Ty` in a query key
    pub def_id_for_ty_in_cycle: Option<DefId>,
}

impl<I> QueryStackFrame<I> {
    #[inline]
    pub fn new(
        info: I,
        dep_kind: DepKind,
        hash: impl FnOnce() -> Hash64,
        def_id: Option<DefId>,
        def_id_for_ty_in_cycle: Option<DefId>,
    ) -> Self {
        Self { info, def_id, dep_kind, hash: hash(), def_id_for_ty_in_cycle }
    }

    fn lift<Qcx: QueryContext<QueryInfo = I>>(
        &self,
        qcx: Qcx,
    ) -> QueryStackFrame<QueryStackFrameExtra> {
        QueryStackFrame {
            info: qcx.lift_query_info(&self.info),
            dep_kind: self.dep_kind,
            hash: self.hash,
            def_id: self.def_id,
            def_id_for_ty_in_cycle: self.def_id_for_ty_in_cycle,
        }
    }
}

#[derive(Clone, Debug)]
pub struct QueryStackFrameExtra {
    pub description: String,
    span: Option<Span>,
    pub def_kind: Option<DefKind>,
}

impl QueryStackFrameExtra {
    #[inline]
    pub fn new(description: String, span: Option<Span>, def_kind: Option<DefKind>) -> Self {
        Self { description, span, def_kind }
    }

    // FIXME(eddyb) Get more valid `Span`s on queries.
    #[inline]
    pub fn default_span(&self, span: Span) -> Span {
        if !span.is_dummy() {
            return span;
        }
        self.span.unwrap_or(span)
    }
}

/// Track a 'side effect' for a particular query.
/// This is used to hold a closure which can create `QueryStackFrameExtra`.
#[derive(Clone)]
pub struct QueryStackDeferred<'tcx> {
    _dummy: PhantomData<&'tcx ()>,

    // `extract` may contain references to 'tcx, but we can't tell drop checking that it won't
    // access it in the destructor.
    extract: Arc<dyn Fn() -> QueryStackFrameExtra + DynSync + DynSend>,
}

impl<'tcx> QueryStackDeferred<'tcx> {
    pub fn new<C: Copy + DynSync + DynSend + 'tcx>(
        context: C,
        extract: fn(C) -> QueryStackFrameExtra,
    ) -> Self {
        let extract: Arc<dyn Fn() -> QueryStackFrameExtra + DynSync + DynSend + 'tcx> =
            Arc::new(move || extract(context));
        // SAFETY: The `extract` closure does not access 'tcx in its destructor as the only
        // captured variable is `context` which is Copy and cannot have a destructor.
        Self { _dummy: PhantomData, extract: unsafe { transmute(extract) } }
    }

    pub fn extract(&self) -> QueryStackFrameExtra {
        (self.extract)()
    }
}

impl<'tcx> Debug for QueryStackDeferred<'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("QueryStackDeferred")
    }
}

/// Tracks 'side effects' for a particular query.
/// This struct is saved to disk along with the query result,
/// and loaded from disk if we mark the query as green.
/// This allows us to 'replay' changes to global state
/// that would otherwise only occur if we actually
/// executed the query method.
///
/// Each side effect gets an unique dep node index which is added
/// as a dependency of the query which had the effect.
#[derive(Debug, Encodable, Decodable)]
pub enum QuerySideEffect {
    /// Stores a diagnostic emitted during query execution.
    /// This diagnostic will be re-emitted if we mark
    /// the query as green, as that query will have the side
    /// effect dep node as a dependency.
    Diagnostic(DiagInner),
}

pub trait QueryContext: HasDepContext {
    type QueryInfo: Clone;

    /// Gets a jobserver reference which is used to release then acquire
    /// a token while waiting on a query.
    fn jobserver_proxy(&self) -> &Proxy;

    fn next_job_id(self) -> QueryJobId;

    /// Get the query information from the TLS context.
    fn current_query_job(self) -> Option<QueryJobId>;

    fn collect_active_jobs(self) -> Result<QueryMap<Self::QueryInfo>, QueryMap<Self::QueryInfo>>;

    fn lift_query_info(self, info: &Self::QueryInfo) -> QueryStackFrameExtra;

    /// Load a side effect associated to the node in the previous session.
    fn load_side_effect(
        self,
        prev_dep_node_index: SerializedDepNodeIndex,
    ) -> Option<QuerySideEffect>;

    /// Register a side effect for the given node, for use in next session.
    fn store_side_effect(self, dep_node_index: DepNodeIndex, side_effect: QuerySideEffect);

    /// Executes a job by changing the `ImplicitCtxt` to point to the
    /// new query job while it executes.
    fn start_query<R>(self, token: QueryJobId, depth_limit: bool, compute: impl FnOnce() -> R)
    -> R;

    fn depth_limit_error(self, job: QueryJobId);
}
