// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::BodyId;
use hir::def_id::DefId;
use hir::map::Map;
use session::Session;
use std::fmt::Debug;
use syntax::ast::NodeId;
use ty::TyCtxt;

use super::{DepGraph, DepNode};

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
impl<'a, 'gcx, 'tcx> DepGraphSafe for TyCtxt<'a, 'gcx, 'tcx> {
}

impl<'a, T> DepGraphSafe for &'a T
    where T: DepGraphSafe
{
}

/// The session gives access to lots of state, but it generates read events.
impl DepGraphSafe for Session {
}

/// The map gives access to lots of state, but it generates read events.
impl<'hir> DepGraphSafe for Map<'hir> {
}

/// The dep-graph better be safe to thread around =)
impl DepGraphSafe for DepGraph {
}

/// DepNodes do not give access to anything in particular, other than
/// def-ids.
impl<D> DepGraphSafe for DepNode<D>
    where D: DepGraphSafe + Clone + Debug
{
}

/// Tuples make it easy to build up state.
impl<A, B> DepGraphSafe for (A, B)
    where A: DepGraphSafe, B: DepGraphSafe
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
