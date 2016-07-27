// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Calculation of a Strict Version Hash for crates.  For a length
//! comment explaining the general idea, see `librustc/middle/svh.rs`.

use syntax::attr::AttributeMethods;
use std::hash::{Hash, SipHasher, Hasher};
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId};
use rustc::hir::map::{NodeItem, NodeForeignItem};
use rustc::hir::svh::Svh;
use rustc::ty::TyCtxt;
use rustc::hir::intravisit::{self, Visitor};

use self::svh_visitor::StrictVersionHashVisitor;

mod svh_visitor;

pub trait SvhCalculate {
    /// Calculate the SVH for an entire krate.
    fn calculate_krate_hash(self) -> Svh;

    /// Calculate the SVH for a particular item.
    fn calculate_item_hash(self, def_id: DefId) -> u64;
}

impl<'a, 'tcx> SvhCalculate for TyCtxt<'a, 'tcx, 'tcx> {
    fn calculate_krate_hash(self) -> Svh {
        // FIXME (#14132): This is better than it used to be, but it still not
        // ideal. We now attempt to hash only the relevant portions of the
        // Crate AST as well as the top-level crate attributes. (However,
        // the hashing of the crate attributes should be double-checked
        // to ensure it is not incorporating implementation artifacts into
        // the hash that are not otherwise visible.)

        let crate_disambiguator = self.sess.local_crate_disambiguator();
        let krate = self.map.krate();

        // FIXME: this should use SHA1, not SipHash. SipHash is not built to
        //        avoid collisions.
        let mut state = SipHasher::new();
        debug!("state: {:?}", state);

        // FIXME(#32753) -- at (*) we `to_le` for endianness, but is
        // this enough, and does it matter anyway?
        "crate_disambiguator".hash(&mut state);
        crate_disambiguator.len().to_le().hash(&mut state); // (*)
        crate_disambiguator.hash(&mut state);

        debug!("crate_disambiguator: {:?}", crate_disambiguator);
        debug!("state: {:?}", state);

        {
            let mut visit = StrictVersionHashVisitor::new(&mut state, self);
            krate.visit_all_items(&mut visit);
        }

        // FIXME (#14132): This hash is still sensitive to e.g. the
        // spans of the crate Attributes and their underlying
        // MetaItems; we should make ContentHashable impl for those
        // types and then use hash_content.  But, since all crate
        // attributes should appear near beginning of the file, it is
        // not such a big deal to be sensitive to their spans for now.
        //
        // We hash only the MetaItems instead of the entire Attribute
        // to avoid hashing the AttrId
        for attr in &krate.attrs {
            debug!("krate attr {:?}", attr);
            attr.meta().hash(&mut state);
        }

        Svh::new(state.finish())
    }

    fn calculate_item_hash(self, def_id: DefId) -> u64 {
        assert!(def_id.is_local());

        debug!("calculate_item_hash(def_id={:?})", def_id);

        let mut state = SipHasher::new();

        {
            let mut visit = StrictVersionHashVisitor::new(&mut state, self);
            if def_id.index == CRATE_DEF_INDEX {
                // the crate root itself is not registered in the map
                // as an item, so we have to fetch it this way
                let krate = self.map.krate();
                intravisit::walk_crate(&mut visit, krate);
            } else {
                let node_id = self.map.as_local_node_id(def_id).unwrap();
                match self.map.find(node_id) {
                    Some(NodeItem(item)) => visit.visit_item(item),
                    Some(NodeForeignItem(item)) => visit.visit_foreign_item(item),
                    r => bug!("calculate_item_hash: expected an item for node {} not {:?}",
                              node_id, r),
                }
            }
        }

        let hash = state.finish();

        debug!("calculate_item_hash: def_id={:?} hash={:?}", def_id, hash);

        hash
    }
}
