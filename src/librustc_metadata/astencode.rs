// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::intravisit::{Visitor, NestedVisitorMap};

use isolated_encoder::IsolatedEncoder;
use schema::*;

use rustc::hir;
use rustc::ty::{self, TyCtxt};

use rustc::ich::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};

#[derive(RustcEncodable, RustcDecodable)]
pub struct Ast<'tcx> {
    pub body: Lazy<hir::Body>,
    pub tables: Lazy<ty::TypeckTables<'tcx>>,
    pub nested_bodies: LazySeq<hir::Body>,
    pub rvalue_promotable_to_static: bool,
    pub stable_bodies_hash: Fingerprint,
}

impl_stable_hash_for!(struct Ast<'tcx> {
    body,
    tables,
    nested_bodies,
    rvalue_promotable_to_static,
    stable_bodies_hash
});

impl<'a, 'b, 'tcx> IsolatedEncoder<'a, 'b, 'tcx> {
    pub fn encode_body(&mut self, body_id: hir::BodyId) -> Lazy<Ast<'tcx>> {
        let body = self.tcx.hir.body(body_id);

        // In order to avoid having to hash hir::Bodies from extern crates, we
        // hash them here, during export, and store the hash with metadata.
        let stable_bodies_hash = {
            let mut hcx = self.tcx.create_stable_hashing_context();
            let mut hasher = StableHasher::new();

            hcx.while_hashing_hir_bodies(true, |hcx| {
                hcx.while_hashing_spans(false, |hcx| {
                    body.hash_stable(hcx, &mut hasher);
                });
            });

            hasher.finish()
        };

        let lazy_body = self.lazy(body);
        let tables = self.tcx.body_tables(body_id);
        let lazy_tables = self.lazy(tables);

        let mut visitor = NestedBodyCollector {
            tcx: self.tcx,
            bodies_found: Vec::new(),
        };
        visitor.visit_body(body);
        let lazy_nested_bodies = self.lazy_seq_ref_from_slice(&visitor.bodies_found);

        let rvalue_promotable_to_static =
            self.tcx.rvalue_promotable_to_static.borrow()[&body.value.id];

        self.lazy(&Ast {
            body: lazy_body,
            tables: lazy_tables,
            nested_bodies: lazy_nested_bodies,
            rvalue_promotable_to_static,
            stable_bodies_hash,
        })
    }
}

struct NestedBodyCollector<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    bodies_found: Vec<&'tcx hir::Body>,
}

impl<'a, 'tcx: 'a> Visitor<'tcx> for NestedBodyCollector<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let body = self.tcx.hir.body(body);
        self.bodies_found.push(body);
        self.visit_body(body);
    }
}
