// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Single-Entry-Multiple-Exit regions: i.e. the definition of a lifetime.

use middle::loop_analysis::LoopAnalysis;
use middle::ty::ctxt;

use syntax::ast::NodeId;

#[deriving(PartialEq, Eq, Hash, Clone, Show, Decodable, Encodable)]
pub struct SemeRegion {
    pub entry: NodeId,
    pub exits: Vec<NodeId>,
}

impl SemeRegion {
    pub fn new(entry: NodeId, mut exits: Vec<NodeId>) -> SemeRegion {
        exits.sort();
        SemeRegion {
            entry: entry,
            exits: exits,
        }
    }

    pub fn dummy() -> SemeRegion {
        SemeRegion::new(-1, Vec::new())
    }

    /// Creates the SEME region corresponding to a scope, with all of its
    /// exits.
    pub fn from_scope(tcx: &ctxt, loop_analysis: &LoopAnalysis, id: NodeId)
                      -> SemeRegion {
        SemeRegion::new(id, loop_analysis.scope_exits(tcx, id))
    }

    /// Returns the ID of the scope that encompasses the whole of this SEME
    /// region. This is, obviously, a lossy conversion.
    pub fn lub_scope(&self, tcx: &ctxt) -> NodeId {
        let mut lub = self.entry;
        for exit in self.exits.iter() {
            match tcx.region_maps.nearest_common_ancestor(lub, *exit) {
                Some(new_lub) => lub = new_lub,
                None => {
                    tcx.sess.bug("SemeRegion::to_block(): didn't find a \
                                  nearest common ancestor")
                }
            }
        }
        lub
    }

    // FIXME(pcwalton): This is imprecise; fill this in.
    pub fn lub(&self,
               other: &SemeRegion,
               tcx: &ctxt,
               loop_analysis: &LoopAnalysis)
               -> Option<SemeRegion> {
        let (this_id, other_id) = (self.lub_scope(tcx), other.lub_scope(tcx));
        match tcx.region_maps.nearest_common_ancestor(this_id, other_id) {
            Some(ancestor_id) => {
                return Some(SemeRegion::from_scope(tcx,
                                                    loop_analysis,
                                                    ancestor_id))
            }
            None => return None,
        }
    }

    // FIXME(pcwalton): This is imprecise; fill this in.
    // FIXME(pcwalton): This can be wrong if SEME regions don't span entire
    // scopes.
    pub fn glb(&self,
               other: &SemeRegion,
               tcx: &ctxt,
               loop_analysis: &LoopAnalysis)
               -> Option<SemeRegion> {
        if *self == *other {
            return Some((*self).clone())
        }

        let (this_id, other_id) = (self.lub_scope(tcx), other.lub_scope(tcx));
        match tcx.region_maps.nearest_common_ancestor(this_id, other_id) {
            Some(ancestor_id) if this_id == ancestor_id => {
                Some(SemeRegion::from_scope(tcx, loop_analysis, other_id))
            }
            Some(ancestor_id) if other_id == ancestor_id => {
                Some(SemeRegion::from_scope(tcx, loop_analysis, this_id))
            }
            Some(_) | None => None,
        }
    }
}

