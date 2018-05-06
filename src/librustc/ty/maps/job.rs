// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::sync::{Lock, Lrc};
use syntax_pos::Span;
use ty::tls;
use ty::maps::Query;
use ty::maps::plumbing::CycleError;
use ty::context::TyCtxt;
use errors::Diagnostic;

/// Indicates the state of a query for a given key in a query map
pub(super) enum QueryResult<'tcx> {
    /// An already executing query. The query job can be used to await for its completion
    Started(Lrc<QueryJob<'tcx>>),

    /// The query panicked. Queries trying to wait on this will raise a fatal error / silently panic
    Poisoned,
}

/// A span and a query key
#[derive(Clone, Debug)]
pub struct QueryInfo<'tcx> {
    /// The span for a reason this query was required
    pub span: Span,
    pub query: Query<'tcx>,
}

/// A object representing an active query job.
pub struct QueryJob<'tcx> {
    pub info: QueryInfo<'tcx>,

    /// The parent query job which created this job and is implicitly waiting on it.
    pub parent: Option<Lrc<QueryJob<'tcx>>>,

    /// Diagnostic messages which are emitted while the query executes
    pub diagnostics: Lock<Vec<Diagnostic>>,
}

impl<'tcx> QueryJob<'tcx> {
    /// Creates a new query job
    pub fn new(info: QueryInfo<'tcx>, parent: Option<Lrc<QueryJob<'tcx>>>) -> Self {
        QueryJob {
            diagnostics: Lock::new(Vec::new()),
            info,
            parent,
        }
    }

    /// Awaits for the query job to complete.
    ///
    /// For single threaded rustc there's no concurrent jobs running, so if we are waiting for any
    /// query that means that there is a query cycle, thus this always running a cycle error.
    pub(super) fn await<'lcx>(
        &self,
        tcx: TyCtxt<'_, 'tcx, 'lcx>,
        span: Span,
    ) -> Result<(), CycleError<'tcx>> {
        // Get the current executing query (waiter) and find the waitee amongst its parents
        let mut current_job = tls::with_related_context(tcx, |icx| icx.query.clone());
        let mut cycle = Vec::new();

        while let Some(job) = current_job {
            cycle.insert(0, job.info.clone());

            if &*job as *const _ == self as *const _ {
                // This is the end of the cycle
                // The span entry we included was for the usage
                // of the cycle itself, and not part of the cycle
                // Replace it with the span which caused the cycle to form
                cycle[0].span = span;
                // Find out why the cycle itself was used
                let usage = job.parent.as_ref().map(|parent| {
                    (job.info.span, parent.info.query.clone())
                });
                return Err(CycleError { usage, cycle });
            }

            current_job = job.parent.clone();
        }

        panic!("did not find a cycle")
    }

    /// Signals to waiters that the query is complete.
    ///
    /// This does nothing for single threaded rustc,
    /// as there are no concurrent jobs which could be waiting on us
    pub fn signal_complete(&self) {}
}
