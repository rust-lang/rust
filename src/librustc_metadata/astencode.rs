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

use common as c;
use cstore;

use decoder::DecodeContext;
use encoder::EncodeContext;

use middle::cstore::{InlinedItem, InlinedItemRef};
use rustc::ty::adjustment;
use rustc::hir::def;
use rustc::hir::def_id::DefId;
use rustc::ty::{self, TyCtxt};

use syntax::ast;

use rbml::reader;
use rbml;
use rustc_serialize::{Decodable, Encodable};

// ______________________________________________________________________
// Top-level methods.

pub fn encode_inlined_item(ecx: &mut EncodeContext, ii: InlinedItemRef) {
    ecx.tag(c::tag_ast, |ecx| {
        ecx.tag(c::tag_id_range, |ecx| {
            let mut visitor = IdRangeComputingVisitor::new();
            match ii {
                InlinedItemRef::Item(_, i) => visitor.visit_item(i),
                InlinedItemRef::TraitItem(_, ti) => visitor.visit_trait_item(ti),
                InlinedItemRef::ImplItem(_, ii) => visitor.visit_impl_item(ii)
            }
            visitor.result().encode(&mut ecx.opaque()).unwrap()
        });

        ecx.tag(c::tag_tree, |ecx| ii.encode(ecx).unwrap());

        ecx.tag(c::tag_table, |ecx| {
            let mut visitor = SideTableEncodingIdVisitor {
                ecx: ecx
            };
            match ii {
                InlinedItemRef::Item(_, i) => visitor.visit_item(i),
                InlinedItemRef::TraitItem(_, ti) => visitor.visit_trait_item(ti),
                InlinedItemRef::ImplItem(_, ii) => visitor.visit_impl_item(ii)
            }
        });
    });
}

/// Decodes an item from its AST in the cdata's metadata and adds it to the
/// ast-map.
pub fn decode_inlined_item<'a, 'tcx>(cdata: &cstore::CrateMetadata,
                                     tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     parent_def_path: ast_map::DefPath,
                                     parent_did: DefId,
                                     ast_doc: rbml::Doc,
                                     orig_did: DefId)
                                     -> &'tcx InlinedItem {
    debug!("> Decoding inlined fn: {:?}", tcx.item_path_str(orig_did));
    let from_id_range = {
        let decoder = &mut ast_doc.get(c::tag_id_range).opaque();
        IdRange {
            min: ast::NodeId::from_u32(u32::decode(decoder).unwrap()),
            max: ast::NodeId::from_u32(u32::decode(decoder).unwrap())
        }
    };
    let mut dcx = DecodeContext::new(tcx, cdata, from_id_range,
                                     ast_doc.get(c::tag_tree));
    let ii = InlinedItem::decode(&mut dcx).unwrap();

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

    decode_side_tables(&mut dcx, ast_doc);

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

    fn id(&mut self, id: ast::NodeId) {
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

fn encode_side_tables_for_id(ecx: &mut EncodeContext, id: ast::NodeId) {
    let tcx = ecx.tcx;

    debug!("Encoding side tables for id {}", id);

    if let Some(def) = tcx.expect_def_or_none(id) {
        ecx.tag(c::tag_table_def, |ecx| {
            ecx.id(id);
            def.encode(ecx).unwrap();
        })
    }

    if let Some(ty) = tcx.node_types().get(&id) {
        ecx.tag(c::tag_table_node_type, |ecx| {
            ecx.id(id);
            ty.encode(ecx).unwrap();
        })
    }

    if let Some(item_substs) = tcx.tables.borrow().item_substs.get(&id) {
        ecx.tag(c::tag_table_item_subst, |ecx| {
            ecx.id(id);
            item_substs.substs.encode(ecx).unwrap();
        })
    }

    if let Some(fv) = tcx.freevars.borrow().get(&id) {
        ecx.tag(c::tag_table_freevars, |ecx| {
            ecx.id(id);
            fv.encode(ecx).unwrap();
        });

        for freevar in fv {
            ecx.tag(c::tag_table_upvar_capture_map, |ecx| {
                ecx.id(id);

                let def_id = freevar.def.def_id();
                let var_id = tcx.map.as_local_node_id(def_id).unwrap();
                let upvar_id = ty::UpvarId {
                    var_id: var_id,
                    closure_expr_id: id
                };
                let upvar_capture = tcx.tables
                                       .borrow()
                                       .upvar_capture_map
                                       .get(&upvar_id)
                                       .unwrap()
                                       .clone();
                var_id.encode(ecx).unwrap();
                upvar_capture.encode(ecx).unwrap();
            })
        }
    }

    let method_call = ty::MethodCall::expr(id);
    if let Some(method) = tcx.tables.borrow().method_map.get(&method_call) {
        ecx.tag(c::tag_table_method_map, |ecx| {
            ecx.id(id);
            method_call.autoderef.encode(ecx).unwrap();
            method.encode(ecx).unwrap();
        })
    }

    if let Some(adjustment) = tcx.tables.borrow().adjustments.get(&id) {
        match *adjustment {
            adjustment::AdjustDerefRef(ref adj) => {
                for autoderef in 0..adj.autoderefs {
                    let method_call = ty::MethodCall::autoderef(id, autoderef as u32);
                    if let Some(method) = tcx.tables.borrow().method_map.get(&method_call) {
                        ecx.tag(c::tag_table_method_map, |ecx| {
                            ecx.id(id);
                            method_call.autoderef.encode(ecx).unwrap();
                            method.encode(ecx).unwrap();
                        })
                    }
                }
            }
            _ => {}
        }

        ecx.tag(c::tag_table_adjustments, |ecx| {
            ecx.id(id);
            adjustment.encode(ecx).unwrap();
        })
    }

    if let Some(cast_kind) = tcx.cast_kinds.borrow().get(&id) {
        ecx.tag(c::tag_table_cast_kinds, |ecx| {
            ecx.id(id);
            cast_kind.encode(ecx).unwrap()
        })
    }

    if let Some(qualif) = tcx.const_qualif_map.borrow().get(&id) {
        ecx.tag(c::tag_table_const_qualif, |ecx| {
            ecx.id(id);
            qualif.encode(ecx).unwrap()
        })
    }
}

fn decode_side_tables<'a, 'tcx>(dcx: &mut DecodeContext<'a, 'tcx>,
                                ast_doc: rbml::Doc<'a>) {
    for (tag, entry_doc) in reader::docs(ast_doc.get(c::tag_table)) {
        dcx.rbml_r = reader::Decoder::new(entry_doc);
        let id = Decodable::decode(dcx).unwrap();
        debug!("decode_side_tables: entry for id={}, tag=0x{:x}", id, tag);
        match tag {
            c::tag_table_def => {
                let def = Decodable::decode(dcx).unwrap();
                dcx.tcx.def_map.borrow_mut().insert(id, def::PathResolution::new(def));
            }
            c::tag_table_node_type => {
                let ty = Decodable::decode(dcx).unwrap();
                dcx.tcx.node_type_insert(id, ty);
            }
            c::tag_table_item_subst => {
                let item_substs = Decodable::decode(dcx).unwrap();
                dcx.tcx.tables.borrow_mut().item_substs.insert(id, item_substs);
            }
            c::tag_table_freevars => {
                let fv_info = Decodable::decode(dcx).unwrap();
                dcx.tcx.freevars.borrow_mut().insert(id, fv_info);
            }
            c::tag_table_upvar_capture_map => {
                let upvar_id = ty::UpvarId {
                    var_id: Decodable::decode(dcx).unwrap(),
                    closure_expr_id: id
                };
                let ub = Decodable::decode(dcx).unwrap();
                dcx.tcx.tables.borrow_mut().upvar_capture_map.insert(upvar_id, ub);
            }
            c::tag_table_method_map => {
                let method_call = ty::MethodCall {
                    expr_id: id,
                    autoderef: Decodable::decode(dcx).unwrap()
                };
                let method = Decodable::decode(dcx).unwrap();
                dcx.tcx.tables.borrow_mut().method_map.insert(method_call, method);
            }
            c::tag_table_adjustments => {
                let adj = Decodable::decode(dcx).unwrap();
                dcx.tcx.tables.borrow_mut().adjustments.insert(id, adj);
            }
            c::tag_table_cast_kinds => {
                let cast_kind = Decodable::decode(dcx).unwrap();
                dcx.tcx.cast_kinds.borrow_mut().insert(id, cast_kind);
            }
            c::tag_table_const_qualif => {
                let qualif = Decodable::decode(dcx).unwrap();
                dcx.tcx.const_qualif_map.borrow_mut().insert(id, qualif);
            }
            _ => {
                bug!("unknown tag found in side tables: 0x{:x}", tag);
            }
        }
    }
}
