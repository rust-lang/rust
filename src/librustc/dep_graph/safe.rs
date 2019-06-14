//! The `DepGraphSafe` trait

use crate::hir::BodyId;
use crate::hir::def_id::DefId;
use syntax::ast::NodeId;
use crate::ty::TyCtxt;

/// The `DepGraphSafe` trait is used to specify what kinds of values
/// are safe to "leak" into a task. The idea is that this should be
/// only be implemented for things like the tcx as well as various id
/// types, which will create reads in the dep-graph whenever the trait
/// loads anything that might depend on the input program.
pub trait DepGraphSafe {
}

/// A `BodyId` on its own doesn't give access to any particular state.
/// You must fetch the state from the various maps or generate
/// on-demand queries, all of which create reads.
impl DepGraphSafe for BodyId {
}

/// A `NodeId` on its own doesn't give access to any particular state.
/// You must fetch the state from the various maps or generate
/// on-demand queries, all of which create reads.
impl DepGraphSafe for NodeId {
}

/// A `DefId` on its own doesn't give access to any particular state.
/// You must fetch the state from the various maps or generate
/// on-demand queries, all of which create reads.
impl DepGraphSafe for DefId {
}

/// The type context itself can be used to access all kinds of tracked
/// state, but those accesses should always generate read events.
impl<'tcx> DepGraphSafe for TyCtxt<'tcx> {}

/// Tuples make it easy to build up state.
impl<A, B> DepGraphSafe for (A, B)
    where A: DepGraphSafe, B: DepGraphSafe
{
}

/// Shared ref to dep-graph-safe stuff should still be dep-graph-safe.
impl<'a, A> DepGraphSafe for &'a A
    where A: DepGraphSafe,
{
}

/// Mut ref to dep-graph-safe stuff should still be dep-graph-safe.
impl<'a, A> DepGraphSafe for &'a mut A
    where A: DepGraphSafe,
{
}


/// No data here! :)
impl DepGraphSafe for () {
}

/// A convenient override that lets you pass arbitrary state into a
/// task. Every use should be accompanied by a comment explaining why
/// it makes sense (or how it could be refactored away in the future).
pub struct AssertDepGraphSafe<T>(pub T);

impl<T> DepGraphSafe for AssertDepGraphSafe<T> {
}
