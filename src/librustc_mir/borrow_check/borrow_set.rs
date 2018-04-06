// Copyright 2012-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dataflow::indexes::BorrowIndex;
use rustc::mir::{self, Location};
use rustc::ty::{Region, RegionKind};
use rustc::util::nodemap::{FxHashMap, FxHashSet};
use rustc_data_structures::indexed_vec::IndexVec;
use std::fmt;
use syntax_pos::Span;

crate struct BorrowSet<'tcx> {
    /// The fundamental map relating bitvector indexes to the borrows
    /// in the MIR.
    crate borrows: IndexVec<BorrowIndex, BorrowData<'tcx>>,

    /// Each borrow is also uniquely identified in the MIR by the
    /// `Location` of the assignment statement in which it appears on
    /// the right hand side; we map each such location to the
    /// corresponding `BorrowIndex`.
    crate location_map: FxHashMap<Location, BorrowIndex>,

    /// Locations which activate borrows.
    /// NOTE: A given location may activate more than one borrow in the future
    /// when more general two-phase borrow support is introduced, but for now we
    /// only need to store one borrow index
    crate activation_map: FxHashMap<Location, FxHashSet<BorrowIndex>>,

    /// Every borrow has a region; this maps each such regions back to
    /// its borrow-indexes.
    crate region_map: FxHashMap<Region<'tcx>, FxHashSet<BorrowIndex>>,

    /// Map from local to all the borrows on that local
    crate local_map: FxHashMap<mir::Local, FxHashSet<BorrowIndex>>,

    /// Maps regions to their corresponding source spans
    /// Only contains ReScope()s as keys
    crate region_span_map: FxHashMap<RegionKind, Span>,
}

#[derive(Debug)]
crate struct BorrowData<'tcx> {
    /// Location where the borrow reservation starts.
    /// In many cases, this will be equal to the activation location but not always.
    crate reserve_location: Location,
    /// Location where the borrow is activated. None if this is not a
    /// 2-phase borrow.
    crate activation_location: Option<Location>,
    /// What kind of borrow this is
    crate kind: mir::BorrowKind,
    /// The region for which this borrow is live
    crate region: Region<'tcx>,
    /// Place from which we are borrowing
    crate borrowed_place: mir::Place<'tcx>,
    /// Place to which the borrow was stored
    crate assigned_place: mir::Place<'tcx>,
}

impl<'tcx> fmt::Display for BorrowData<'tcx> {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result {
        let kind = match self.kind {
            mir::BorrowKind::Shared => "",
            mir::BorrowKind::Unique => "uniq ",
            mir::BorrowKind::Mut { .. } => "mut ",
        };
        let region = format!("{}", self.region);
        let region = if region.len() > 0 { format!("{} ", region) } else { region };
        write!(w, "&{}{}{:?}", region, kind, self.borrowed_place)
    }
}

