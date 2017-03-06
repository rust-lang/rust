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
use syntax::ast::NodeId;
use ty::TyCtxt;

use super::graph::DepGraph;

/// The `DepGraphSafe` auto trait is used to specify what kinds of
/// values are safe to "leak" into a task.  The idea is that this
/// should be only be implemented for things like the tcx, which will
/// create reads in the dep-graph whenever the trait loads anything
/// that might depend on the input program.
pub trait DepGraphSafe {
    fn read(&self, graph: &DepGraph);
}

impl DepGraphSafe for BodyId {
    fn read(&self, _graph: &DepGraph) {
        // a BodyId on its own doesn't give access to any particular state
    }
}

impl DepGraphSafe for NodeId {
    fn read(&self, _graph: &DepGraph) {
        // a DefId doesn't give any particular state
    }
}

impl DepGraphSafe for DefId {
    fn read(&self, _graph: &DepGraph) {
        // a DefId doesn't give any particular state
    }
}

impl<'a, 'gcx, 'tcx> DepGraphSafe for TyCtxt<'a, 'gcx, 'tcx> {
    fn read(&self, _graph: &DepGraph) {
    }
}

impl<A, B> DepGraphSafe for (A, B)
    where A: DepGraphSafe, B: DepGraphSafe
{
    fn read(&self, graph: &DepGraph) {
        self.0.read(graph);
        self.1.read(graph);
    }
}

impl DepGraphSafe for () {
    fn read(&self, _graph: &DepGraph) {
    }
}

/// A convenient override. We should phase out usage of this over
/// time.
pub struct AssertDepGraphSafe<T>(pub T);
impl<T> DepGraphSafe for AssertDepGraphSafe<T> {
    fn read(&self, _graph: &DepGraph) {
    }
}
