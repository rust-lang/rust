// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(warnings)]

use std::mem;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use rustc_data_structures::sync::{Lock, LockGuard, Lrc};
use syntax_pos::Span;
use ty::tls;
use ty::maps::Query;
use ty::maps::plumbing::CycleError;
use ty::context::TyCtxt;
use errors::Diagnostic;
use std::process;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::collections::HashSet;

pub struct PoisonedJob;

#[derive(Clone, Debug)]
pub struct StackEntry<'tcx> {
    pub span: Span,
    pub query: Query<'tcx>,
}

pub struct QueryJob<'tcx> {
    pub entry: StackEntry<'tcx>,
    pub parent: Option<Lrc<QueryJob<'tcx>>>,
    pub track_diagnostics: bool,
    pub diagnostics: Lock<Vec<Diagnostic>>,
}

impl<'tcx> QueryJob<'tcx> {
    pub fn new(
        entry: StackEntry<'tcx>,
        track_diagnostics: bool,
        parent: Option<Lrc<QueryJob<'tcx>>>,
    ) -> Self {
        QueryJob {
            track_diagnostics,
            diagnostics: Lock::new(Vec::new()),
            entry,
            parent,
        }
    }

    pub(super) fn await<'lcx>(
        &self,
        tcx: TyCtxt<'_, 'tcx, 'lcx>,
        span: Span,
    ) -> Result<(), CycleError<'tcx>> {
        // The query is already executing, so this must be a cycle for single threaded rustc,
        // so we find the cycle and return it

        let mut current_job = tls::with_related_context(tcx, |icx| icx.query.clone());
        let mut cycle = Vec::new();

        while let Some(job) = current_job {
            cycle.insert(0, job.entry.clone());

            if &*job as *const _ == self as *const _ {
                break;
            }

            current_job = job.parent.clone();
        }

        Err(CycleError { span, cycle })
    }

    pub fn signal_complete(&self) {
        // Signals to waiters that the query is complete.
        // This is a no-op for single threaded rustc
    }
}

pub(super) enum QueryResult<'tcx, T> {
    Started(Lrc<QueryJob<'tcx>>),
    Complete(T),
    Poisoned,
}
