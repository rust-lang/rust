// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::map as ast_map;

use rustc::hir::intravisit::{Visitor, IdRangeComputingVisitor, IdRange, NestedVisitorMap};

use cstore::CrateMetadata;
use encoder::EncodeContext;
use schema::*;

use rustc::hir;
use rustc::hir::def::Def;
use rustc::hir::def_id::DefId;
use rustc::ty::{self, TyCtxt, Ty};

use syntax::ast;

use rustc_serialize::Encodable;

#[derive(RustcEncodable, RustcDecodable)]
pub struct Ast<'tcx> {
    id_range: IdRange,
    body: Lazy<hir::Body>,
    side_tables: LazySeq<(ast::NodeId, TableEntry<'tcx>)>,
    pub nested_bodies: LazySeq<hir::Body>,
    pub rvalue_promotable_to_static: bool,
}

#[derive(RustcEncodable, RustcDecodable)]
enum TableEntry<'tcx> {
    TypeRelativeDef(Def),
    NodeType(Ty<'tcx>),
    ItemSubsts(ty::ItemSubsts<'tcx>),
    Adjustment(ty::adjustment::Adjustment<'tcx>),
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    pub fn encode_body(&mut self, body: hir::BodyId) -> Lazy<Ast<'tcx>> {
        let body = self.tcx.map.body(body);

        let mut id_visitor = IdRangeComputingVisitor::new(&self.tcx.map);
        id_visitor.visit_body(body);

        let body_pos = self.position();
        body.encode(self).unwrap();

        let tables_pos = self.position();
        let tables_count = {
            let mut visitor = SideTableEncodingIdVisitor {
                ecx: self,
                count: 0,
            };
            visitor.visit_body(body);
            visitor.count
        };

        let nested_pos = self.position();
        let nested_count = {
            let mut visitor = NestedBodyEncodingVisitor {
                ecx: self,
                count: 0,
            };
            visitor.visit_body(body);
            visitor.count
        };

        let rvalue_promotable_to_static =
            self.tcx.rvalue_promotable_to_static.borrow()[&body.value.id];

        self.lazy(&Ast {
            id_range: id_visitor.result(),
            body: Lazy::with_position(body_pos),
            side_tables: LazySeq::with_position_and_length(tables_pos, tables_count),
            nested_bodies: LazySeq::with_position_and_length(nested_pos, nested_count),
            rvalue_promotable_to_static: rvalue_promotable_to_static
        })
    }
}

struct SideTableEncodingIdVisitor<'a, 'b: 'a, 'tcx: 'b> {
    ecx: &'a mut EncodeContext<'b, 'tcx>,
    count: usize,
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for SideTableEncodingIdVisitor<'a, 'b, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.ecx.tcx.map)
    }

    fn visit_id(&mut self, id: ast::NodeId) {
        debug!("Encoding side tables for id {}", id);

        let tcx = self.ecx.tcx;
        let mut encode = |entry: Option<TableEntry>| {
            if let Some(entry) = entry {
                (id, entry).encode(self.ecx).unwrap();
                self.count += 1;
            }
        };

        encode(tcx.tables().type_relative_path_defs.get(&id).cloned()
                  .map(TableEntry::TypeRelativeDef));
        encode(tcx.tables().node_types.get(&id).cloned().map(TableEntry::NodeType));
        encode(tcx.tables().item_substs.get(&id).cloned().map(TableEntry::ItemSubsts));
        encode(tcx.tables().adjustments.get(&id).cloned().map(TableEntry::Adjustment));
    }
}

struct NestedBodyEncodingVisitor<'a, 'b: 'a, 'tcx: 'b> {
    ecx: &'a mut EncodeContext<'b, 'tcx>,
    count: usize,
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for NestedBodyEncodingVisitor<'a, 'b, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let body = self.ecx.tcx.map.body(body);
        body.encode(self.ecx).unwrap();
        self.count += 1;

        self.visit_body(body);
    }
}

/// Decodes an item's body from its AST in the cdata's metadata and adds it to the
/// ast-map.
pub fn decode_body<'a, 'tcx>(cdata: &CrateMetadata,
                             tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             def_id: DefId,
                             ast: Ast<'tcx>)
                             -> &'tcx hir::Body {
    debug!("> Decoding inlined fn: {}", tcx.item_path_str(def_id));

    let cnt = ast.id_range.max.as_usize() - ast.id_range.min.as_usize();
    let start = tcx.sess.reserve_node_ids(cnt);
    let id_ranges = [ast.id_range,
                     IdRange {
                         min: start,
                         max: ast::NodeId::new(start.as_usize() + cnt),
                     }];

    for (id, entry) in ast.side_tables.decode((cdata, tcx, id_ranges)) {
        match entry {
            TableEntry::TypeRelativeDef(def) => {
                tcx.tables.borrow_mut().type_relative_path_defs.insert(id, def);
            }
            TableEntry::NodeType(ty) => {
                tcx.tables.borrow_mut().node_types.insert(id, ty);
            }
            TableEntry::ItemSubsts(item_substs) => {
                tcx.tables.borrow_mut().item_substs.insert(id, item_substs);
            }
            TableEntry::Adjustment(adj) => {
                tcx.tables.borrow_mut().adjustments.insert(id, adj);
            }
        }
    }

    let body = ast.body.decode((cdata, tcx, id_ranges));
    ast_map::map_decoded_body(&tcx.map, def_id, body, tcx.sess.next_node_id())
}
