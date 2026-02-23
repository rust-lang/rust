use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::transmute;
use std::sync::Arc;

use rustc_data_structures::sync::{DynSend, DynSync};
use rustc_hashes::Hash64;
use rustc_hir::def::DefKind;
use rustc_span::Span;
use rustc_span::def_id::DefId;

use crate::dep_graph::DepKind;

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
    pub hash: Hash64,
    pub def_id: Option<DefId>,
    /// A def-id that is extracted from a `Ty` in a query key
    pub def_id_for_ty_in_cycle: Option<DefId>,
}

impl<'tcx> QueryStackFrame<QueryStackDeferred<'tcx>> {
    #[inline]
    pub fn new(
        info: QueryStackDeferred<'tcx>,
        dep_kind: DepKind,
        hash: Hash64,
        def_id: Option<DefId>,
        def_id_for_ty_in_cycle: Option<DefId>,
    ) -> Self {
        Self { info, def_id, dep_kind, hash, def_id_for_ty_in_cycle }
    }

    pub fn lift(&self) -> QueryStackFrame<QueryStackFrameExtra> {
        QueryStackFrame {
            info: self.info.extract(),
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
    pub span: Option<Span>,
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
