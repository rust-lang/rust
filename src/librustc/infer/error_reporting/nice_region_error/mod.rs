// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::InferCtxt;
use infer::lexical_region_resolve::RegionResolutionError;
use infer::lexical_region_resolve::RegionResolutionError::*;
use syntax::codemap::Span;
use ty::{self, TyCtxt};
use util::common::ErrorReported;

mod different_lifetimes;
mod find_anon_type;
mod named_anon_conflict;
mod util;

impl<'cx, 'gcx, 'tcx> InferCtxt<'cx, 'gcx, 'tcx> {
    pub fn try_report_nice_region_error(&self, error: &RegionResolutionError<'tcx>) -> bool {
        let (span, sub, sup) = match *error {
            ConcreteFailure(ref origin, sub, sup) => (origin.span(), sub, sup),
            SubSupConflict(_, ref origin, sub, _, sup) => (origin.span(), sub, sup),
            _ => return false, // inapplicable
        };

        if let Some(tables) = self.in_progress_tables {
            let tables = tables.borrow();
            NiceRegionError::new(self.tcx, span, sub, sup, Some(&tables)).try_report().is_some()
        } else {
            NiceRegionError::new(self.tcx, span, sub, sup, None).try_report().is_some()
        }
    }
}

pub struct NiceRegionError<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    span: Span,
    sub: ty::Region<'tcx>,
    sup: ty::Region<'tcx>,
    tables: Option<&'cx ty::TypeckTables<'tcx>>,
}

impl<'cx, 'gcx, 'tcx> NiceRegionError<'cx, 'gcx, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'cx, 'gcx, 'tcx>,
        span: Span,
        sub: ty::Region<'tcx>,
        sup: ty::Region<'tcx>,
        tables: Option<&'cx ty::TypeckTables<'tcx>>,
    ) -> Self {
        Self { tcx, span, sub, sup, tables }
    }

    pub fn try_report(&self) -> Option<ErrorReported> {
        self.try_report_named_anon_conflict()
            .or_else(|| self.try_report_anon_anon_conflict())
    }
}
