// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

use rustc::hir::map as ast_map;

use rustc::hir::intravisit::{Visitor, IdRangeComputingVisitor, IdRange};

use cstore::CrateMetadata;
use decoder::DecodeContext;
use encoder::EncodeContext;

use rustc::middle::cstore::{InlinedItem, InlinedItemRef};
use rustc::hir::def;
use rustc::hir::def_id::DefId;
use rustc::ty::TyCtxt;

use syntax::ast;

use rbml;
use rustc_serialize::{Decodable, Encodable};

// ______________________________________________________________________
// Top-level methods.

pub fn encode_inlined_item(ecx: &mut EncodeContext, ii: InlinedItemRef) {
    ecx.tag(::common::item_tag::ast, |ecx| {
        let mut visitor = IdRangeComputingVisitor::new();
        match ii {
            InlinedItemRef::Item(_, i) => visitor.visit_item(i),
            InlinedItemRef::TraitItem(_, ti) => visitor.visit_trait_item(ti),
            InlinedItemRef::ImplItem(_, ii) => visitor.visit_impl_item(ii)
        }
        visitor.result().encode(ecx).unwrap();

        ii.encode(ecx).unwrap();

        let mut visitor = SideTableEncodingIdVisitor {
            ecx: ecx
        };
        match ii {
            InlinedItemRef::Item(_, i) => visitor.visit_item(i),
            InlinedItemRef::TraitItem(_, ti) => visitor.visit_trait_item(ti),
            InlinedItemRef::ImplItem(_, ii) => visitor.visit_impl_item(ii)
        }
    });
}

/// Decodes an item from its AST in the cdata's metadata and adds it to the
/// ast-map.
pub fn decode_inlined_item<'a, 'tcx>(cdata: &CrateMetadata,
                                     tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     parent_def_path: ast_map::DefPath,
                                     parent_did: DefId,
                                     ast_doc: rbml::Doc,
                                     orig_did: DefId)
                                     -> &'tcx InlinedItem {
    debug!("> Decoding inlined fn: {:?}", tcx.item_path_str(orig_did));
    let dcx = &mut ast_doc.decoder();
    dcx.tcx = Some(tcx);
    dcx.cdata = Some(cdata);
    dcx.from_id_range = IdRange::decode(dcx).unwrap();
    let cnt = dcx.from_id_range.max.as_usize() - dcx.from_id_range.min.as_usize();
    dcx.to_id_range.min = tcx.sess.reserve_node_ids(cnt);
    dcx.to_id_range.max = ast::NodeId::new(dcx.to_id_range.min.as_usize() + cnt);
    let ii = InlinedItem::decode(dcx).unwrap();

    let ii = ast_map::map_decoded_item(&tcx.map,
                                       parent_def_path,
                                       parent_did,
                                       ii,
                                       tcx.sess.next_node_id());

    let item_node_id = match ii {
        &InlinedItem::Item(_, ref i) => i.id,
        &InlinedItem::TraitItem(_, ref ti) => ti.id,
        &InlinedItem::ImplItem(_, ref ii) => ii.id
    };
    let inlined_did = tcx.map.local_def_id(item_node_id);
    tcx.register_item_type(inlined_did, tcx.lookup_item_type(orig_did));

    decode_side_tables(dcx, ast_doc);

    ii
}

// ______________________________________________________________________
// Encoding and decoding the side tables

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn tag<F>(&mut self,
              tag_id: usize,
              f: F) where
        F: FnOnce(&mut Self),
    {
        self.start_tag(tag_id).unwrap();
        f(self);
        self.end_tag().unwrap();
    }

    fn entry(&mut self, table: Table, id: ast::NodeId) {
        table.encode(self).unwrap();
        id.encode(self).unwrap();
    }
}

struct SideTableEncodingIdVisitor<'a, 'b:'a, 'tcx:'b> {
    ecx: &'a mut EncodeContext<'b, 'tcx>,
}

impl<'a, 'b, 'tcx, 'v> Visitor<'v> for SideTableEncodingIdVisitor<'a, 'b, 'tcx> {
    fn visit_id(&mut self, id: ast::NodeId) {
        encode_side_tables_for_id(self.ecx, id)
    }
}

#[derive(RustcEncodable, RustcDecodable, Debug)]
enum Table {
    Def,
    NodeType,
    ItemSubsts,
    Adjustment,
    ConstQualif
}

fn encode_side_tables_for_id(ecx: &mut EncodeContext, id: ast::NodeId) {
    let tcx = ecx.tcx;

    debug!("Encoding side tables for id {}", id);

    if let Some(def) = tcx.expect_def_or_none(id) {
        ecx.entry(Table::Def, id);
        def.encode(ecx).unwrap();
    }

    if let Some(ty) = tcx.node_types().get(&id) {
        ecx.entry(Table::NodeType, id);
        ty.encode(ecx).unwrap();
    }

    if let Some(item_substs) = tcx.tables.borrow().item_substs.get(&id) {
        ecx.entry(Table::ItemSubsts, id);
        item_substs.substs.encode(ecx).unwrap();
    }

    if let Some(adjustment) = tcx.tables.borrow().adjustments.get(&id) {
        ecx.entry(Table::Adjustment, id);
        adjustment.encode(ecx).unwrap();
    }

    if let Some(qualif) = tcx.const_qualif_map.borrow().get(&id) {
        ecx.entry(Table::ConstQualif, id);
        qualif.encode(ecx).unwrap();
    }
}

fn decode_side_tables(dcx: &mut DecodeContext, ast_doc: rbml::Doc) {
    while dcx.opaque.position() < ast_doc.end {
        let table = Decodable::decode(dcx).unwrap();
        let id = Decodable::decode(dcx).unwrap();
        debug!("decode_side_tables: entry for id={}, table={:?}", id, table);
        match table {
            Table::Def => {
                let def = Decodable::decode(dcx).unwrap();
                dcx.tcx().def_map.borrow_mut().insert(id, def::PathResolution::new(def));
            }
            Table::NodeType => {
                let ty = Decodable::decode(dcx).unwrap();
                dcx.tcx().node_type_insert(id, ty);
            }
            Table::ItemSubsts => {
                let item_substs = Decodable::decode(dcx).unwrap();
                dcx.tcx().tables.borrow_mut().item_substs.insert(id, item_substs);
            }
            Table::Adjustment => {
                let adj = Decodable::decode(dcx).unwrap();
                dcx.tcx().tables.borrow_mut().adjustments.insert(id, adj);
            }
            Table::ConstQualif => {
                let qualif = Decodable::decode(dcx).unwrap();
                dcx.tcx().const_qualif_map.borrow_mut().insert(id, qualif);
            }
        }
    }
}
