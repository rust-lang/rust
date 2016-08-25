// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use rustc::mir::repr as mir;

use std::marker::PhantomData;
use super::*;

/// This combinator has the following behaviour:
///
/// * Rewrite the node with the first rewriter.
///   * if the first rewriter replaced the node, 2nd rewriter is used to rewrite the replacement.
///   * otherwise 2nd rewriter is used to rewrite the original node.
pub struct RewriteAndThen<'tcx, T, R1, R2>(R1, R2, PhantomData<(&'tcx (), T)>)
where T: Transfer<'tcx>, R1: Rewrite<'tcx, T>, R2: Rewrite<'tcx, T>;

impl<'tcx, T, R1, R2> RewriteAndThen<'tcx, T, R1, R2>
where T: Transfer<'tcx>, R1: Rewrite<'tcx, T>, R2: Rewrite<'tcx, T>
{
    pub fn new(r1: R1, r2: R2) -> RewriteAndThen<'tcx, T, R1, R2> {
        RewriteAndThen(r1, r2, PhantomData)
    }
}

impl<'tcx, T, R1, R2> Rewrite<'tcx, T> for RewriteAndThen<'tcx, T, R1, R2>
where T: Transfer<'tcx>, R1: Rewrite<'tcx, T>, R2: Rewrite<'tcx, T> {
    fn stmt(&self, stmt: &mir::Statement<'tcx>, fact: &T::Lattice) -> StatementChange<'tcx> {
        let rs = self.0.stmt(stmt, fact);
        match rs {
            StatementChange::Remove => StatementChange::Remove,
            StatementChange::Statement(ns) => self.1.stmt(&ns, fact),
        }
    }

    fn term(&self, term: &mir::Terminator<'tcx>, fact: &T::Lattice) -> TerminatorChange<'tcx> {
        let rt = self.0.term(term, fact);
        match rt {
            TerminatorChange::Terminator(nt) => self.1.term(&nt, fact)
        }
    }
}
