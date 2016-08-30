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
// FIXME: remove this after snapshot, and Results are handled
#![allow(unused_must_use)]

use rustc::hir::map as ast_map;

use rustc::hir;
use rustc::hir::fold;
use rustc::hir::fold::Folder;
use rustc::hir::intravisit::{Visitor, IdRangeComputingVisitor, IdRange};

use common as c;
use cstore;
use decoder;

use decoder::DecodeContext;
use encoder::EncodeContext;

use middle::cstore::{InlinedItem, InlinedItemRef};
use rustc::ty::adjustment;
use rustc::ty::cast;
use middle::const_qualif::ConstQualif;
use rustc::hir::def::{self, Def};
use rustc::hir::def_id::DefId;
use rustc::ty::{self, Ty, TyCtxt};

use syntax::ast;
use syntax::ptr::P;
use syntax_pos;

use std::io::SeekFrom;
use std::io::prelude::*;

use rbml::reader;
use rbml;
use rustc_serialize::{Decodable, Decoder, DecoderHelpers};
use rustc_serialize::{Encodable, EncoderHelpers};

trait tr {
    fn tr(&self, dcx: &DecodeContext) -> Self;
}

// ______________________________________________________________________
// Top-level methods.

pub fn encode_inlined_item(ecx: &mut EncodeContext, ii: InlinedItemRef) {
    let id = match ii {
        InlinedItemRef::Item(_, i) => i.id,
        InlinedItemRef::TraitItem(_, ti) => ti.id,
        InlinedItemRef::ImplItem(_, ii) => ii.id,
    };
    debug!("> Encoding inlined item: {} ({:?})",
           ecx.tcx.node_path_str(id),
           ecx.writer.seek(SeekFrom::Current(0)));

    // Folding could be avoided with a smarter encoder.
    let (ii, expected_id_range) = simplify_ast(ii);
    let id_range = inlined_item_id_range(&ii);
    assert_eq!(expected_id_range, id_range);

    ecx.start_tag(c::tag_ast);

    ecx.start_tag(c::tag_id_range);
    id_range.encode(&mut ecx.opaque());
    ecx.end_tag();

    ecx.start_tag(c::tag_tree);
    ii.encode(ecx);
    ecx.end_tag();

    encode_side_tables_for_ii(ecx, &ii);
    ecx.end_tag();

    debug!("< Encoded inlined fn: {} ({:?})",
           ecx.tcx.node_path_str(id),
           ecx.writer.seek(SeekFrom::Current(0)));
}

impl<'a, 'b, 'tcx> ast_map::FoldOps for &'a DecodeContext<'b, 'tcx> {
    fn new_id(&self, id: ast::NodeId) -> ast::NodeId {
        if id == ast::DUMMY_NODE_ID {
            // Used by ast_map to map the NodeInlinedParent.
            self.tcx.sess.next_node_id()
        } else {
            self.tr_id(id)
        }
    }
    fn new_def_id(&self, def_id: DefId) -> DefId {
        self.tr_def_id(def_id)
    }
    fn new_span(&self, span: syntax_pos::Span) -> syntax_pos::Span {
        self.tr_span(span)
    }
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
    let id_range_doc = ast_doc.get(c::tag_id_range);
    let from_id_range = IdRange::decode(&mut id_range_doc.opaque()).unwrap();
    let mut dcx = DecodeContext::new(tcx, cdata, from_id_range,
                                     ast_doc.get(c::tag_tree));
    let ii = InlinedItem::decode(&mut dcx).unwrap();

    let ii = ast_map::map_decoded_item(&tcx.map,
                                       parent_def_path,
                                       parent_did,
                                       ii,
                                       &dcx);

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
// Enumerating the IDs which appear in an AST

impl<'a, 'tcx> DecodeContext<'a, 'tcx> {
    /// Translates an internal id, meaning a node id that is known to refer to some part of the
    /// item currently being inlined, such as a local variable or argument.  All naked node-ids
    /// that appear in types have this property, since if something might refer to an external item
    /// we would use a def-id to allow for the possibility that the item resides in another crate.
    pub fn tr_id(&self, id: ast::NodeId) -> ast::NodeId {
        // from_id_range should be non-empty
        assert!(!self.from_id_range.empty());
        // Make sure that translating the NodeId will actually yield a
        // meaningful result
        assert!(self.from_id_range.contains(id));

        // Use wrapping arithmetic because otherwise it introduces control flow.
        // Maybe we should just have the control flow? -- aatch
        (id.wrapping_sub(self.from_id_range.min).wrapping_add(self.to_id_range.min))
    }

    /// Translates an EXTERNAL def-id, converting the crate number from the one used in the encoded
    /// data to the current crate numbers..  By external, I mean that it be translated to a
    /// reference to the item in its original crate, as opposed to being translated to a reference
    /// to the inlined version of the item.  This is typically, but not always, what you want,
    /// because most def-ids refer to external things like types or other fns that may or may not
    /// be inlined.  Note that even when the inlined function is referencing itself recursively, we
    /// would want `tr_def_id` for that reference--- conceptually the function calls the original,
    /// non-inlined version, and trans deals with linking that recursive call to the inlined copy.
    pub fn tr_def_id(&self, did: DefId) -> DefId {
        decoder::translate_def_id(self.cdata, did)
    }

    /// Translates a `Span` from an extern crate to the corresponding `Span`
    /// within the local crate's codemap.
    pub fn tr_span(&self, span: syntax_pos::Span) -> syntax_pos::Span {
        decoder::translate_span(self.cdata,
                                self.tcx.sess.codemap(),
                                &self.last_filemap_index,
                                span)
    }
}

impl tr for DefId {
    fn tr(&self, dcx: &DecodeContext) -> DefId {
        dcx.tr_def_id(*self)
    }
}

impl tr for Option<DefId> {
    fn tr(&self, dcx: &DecodeContext) -> Option<DefId> {
        self.map(|d| dcx.tr_def_id(d))
    }
}

impl tr for syntax_pos::Span {
    fn tr(&self, dcx: &DecodeContext) -> syntax_pos::Span {
        dcx.tr_span(*self)
    }
}

// ______________________________________________________________________
// Encoding and decoding the AST itself
//
// When decoding, we have to renumber the AST so that the node ids that
// appear within are disjoint from the node ids in our existing ASTs.
// We also have to adjust the spans: for now we just insert a dummy span,
// but eventually we should add entries to the local codemap as required.

struct NestedItemsDropper {
    id_range: IdRange
}

impl Folder for NestedItemsDropper {

    // The unit tests below run on HIR with NodeIds not properly assigned. That
    // causes an integer overflow. So we just don't track the id_range when
    // building the unit tests.
    #[cfg(not(test))]
    fn new_id(&mut self, id: ast::NodeId) -> ast::NodeId {
        // Record the range of NodeIds we are visiting, so we can do a sanity
        // check later
        self.id_range.add(id);
        id
    }

    fn fold_block(&mut self, blk: P<hir::Block>) -> P<hir::Block> {
        blk.and_then(|hir::Block {id, stmts, expr, rules, span, ..}| {
            let stmts_sans_items = stmts.into_iter().filter_map(|stmt| {
                let use_stmt = match stmt.node {
                    hir::StmtExpr(..) | hir::StmtSemi(..) => true,
                    hir::StmtDecl(ref decl, _) => {
                        match decl.node {
                            hir::DeclLocal(_) => true,
                            hir::DeclItem(_) => false,
                        }
                    }
                };
                if use_stmt {
                    Some(stmt)
                } else {
                    None
                }
            }).collect();
            let blk_sans_items = P(hir::Block {
                stmts: stmts_sans_items,
                expr: expr,
                id: id,
                rules: rules,
                span: span,
            });
            fold::noop_fold_block(blk_sans_items, self)
        })
    }
}

// Produces a simplified copy of the AST which does not include things
// that we do not need to or do not want to export.  For example, we
// do not include any nested items: if these nested items are to be
// inlined, their AST will be exported separately (this only makes
// sense because, in Rust, nested items are independent except for
// their visibility).
//
// As it happens, trans relies on the fact that we do not export
// nested items, as otherwise it would get confused when translating
// inlined items.
fn simplify_ast(ii: InlinedItemRef) -> (InlinedItem, IdRange) {
    let mut fld = NestedItemsDropper {
        id_range: IdRange::max()
    };

    let ii = match ii {
        // HACK we're not dropping items.
        InlinedItemRef::Item(d, i) => {
            InlinedItem::Item(d, P(fold::noop_fold_item(i.clone(), &mut fld)))
        }
        InlinedItemRef::TraitItem(d, ti) => {
            InlinedItem::TraitItem(d, P(fold::noop_fold_trait_item(ti.clone(), &mut fld)))
        }
        InlinedItemRef::ImplItem(d, ii) => {
            InlinedItem::ImplItem(d, P(fold::noop_fold_impl_item(ii.clone(), &mut fld)))
        }
    };

    (ii, fld.id_range)
}

// ______________________________________________________________________
// Encoding and decoding of ast::def

impl tr for Def {
    fn tr(&self, dcx: &DecodeContext) -> Def {
        match *self {
          Def::Fn(did) => Def::Fn(did.tr(dcx)),
          Def::Method(did) => Def::Method(did.tr(dcx)),
          Def::SelfTy(opt_did, impl_id) => {
              // Since the impl_id will never lie within the reserved range of
              // imported NodeIds, it does not make sense to translate it.
              // The result would not make any sense within the importing crate.
              // We also don't allow for impl items to be inlined (just their
              // members), so even if we had a DefId here, we wouldn't be able
              // to do much with it.
              // So, we set the id to DUMMY_NODE_ID. That way we make it
              // explicit that this is no usable NodeId.
              Def::SelfTy(opt_did.map(|did| did.tr(dcx)),
                          impl_id.map(|_| ast::DUMMY_NODE_ID))
          }
          Def::Mod(did) => { Def::Mod(did.tr(dcx)) }
          Def::ForeignMod(did) => { Def::ForeignMod(did.tr(dcx)) }
          Def::Static(did, m) => { Def::Static(did.tr(dcx), m) }
          Def::Const(did) => { Def::Const(did.tr(dcx)) }
          Def::AssociatedConst(did) => Def::AssociatedConst(did.tr(dcx)),
          Def::Local(_, nid) => {
              let nid = dcx.tr_id(nid);
              let did = dcx.tcx.map.local_def_id(nid);
              Def::Local(did, nid)
          }
          Def::Variant(e_did, v_did) => Def::Variant(e_did.tr(dcx), v_did.tr(dcx)),
          Def::Trait(did) => Def::Trait(did.tr(dcx)),
          Def::Enum(did) => Def::Enum(did.tr(dcx)),
          Def::TyAlias(did) => Def::TyAlias(did.tr(dcx)),
          Def::AssociatedTy(trait_did, did) =>
              Def::AssociatedTy(trait_did.tr(dcx), did.tr(dcx)),
          Def::PrimTy(p) => Def::PrimTy(p),
          Def::TyParam(did) => Def::TyParam(did.tr(dcx)),
          Def::Upvar(_, nid1, index, nid2) => {
              let nid1 = dcx.tr_id(nid1);
              let nid2 = dcx.tr_id(nid2);
              let did1 = dcx.tcx.map.local_def_id(nid1);
              Def::Upvar(did1, nid1, index, nid2)
          }
          Def::Struct(did) => Def::Struct(did.tr(dcx)),
          Def::Union(did) => Def::Union(did.tr(dcx)),
          Def::Label(nid) => Def::Label(dcx.tr_id(nid)),
          Def::Err => Def::Err,
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of freevar information

impl<'a, 'tcx> DecodeContext<'a, 'tcx> {
    fn read_freevar_entry(&mut self) -> hir::Freevar {
        hir::Freevar::decode(self).unwrap().tr(self)
    }
}

impl tr for hir::Freevar {
    fn tr(&self, dcx: &DecodeContext) -> hir::Freevar {
        hir::Freevar {
            def: self.def.tr(dcx),
            span: self.span.tr(dcx),
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of MethodCallee

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn encode_method_callee(&mut self,
                            autoderef: u32,
                            method: &ty::MethodCallee<'tcx>) {
        use rustc_serialize::Encoder;

        self.emit_struct("MethodCallee", 4, |this| {
            this.emit_struct_field("autoderef", 0, |this| {
                autoderef.encode(this)
            });
            this.emit_struct_field("def_id", 1, |this| {
                method.def_id.encode(this)
            });
            this.emit_struct_field("ty", 2, |this| {
                method.ty.encode(this)
            });
            this.emit_struct_field("substs", 3, |this| {
                method.substs.encode(this)
            })
        }).unwrap();
    }
}

impl<'a, 'tcx> DecodeContext<'a, 'tcx> {
    fn read_method_callee(&mut self)  -> (u32, ty::MethodCallee<'tcx>) {
        self.read_struct("MethodCallee", 4, |this| {
            let autoderef = this.read_struct_field("autoderef", 0,
                                                   Decodable::decode).unwrap();
            Ok((autoderef, ty::MethodCallee {
                def_id: this.read_struct_field("def_id", 1, |this| {
                    DefId::decode(this).map(|d| d.tr(this))
                }).unwrap(),
                ty: this.read_struct_field("ty", 2, |this| {
                    Ty::decode(this)
                }).unwrap(),
                substs: this.read_struct_field("substs", 3, |this| {
                    Decodable::decode(this)
                }).unwrap()
            }))
        }).unwrap()
    }
}

// ______________________________________________________________________
// Encoding and decoding the side tables

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn emit_upvar_capture(&mut self, capture: &ty::UpvarCapture<'tcx>) {
        use rustc_serialize::Encoder;

        self.emit_enum("UpvarCapture", |this| {
            match *capture {
                ty::UpvarCapture::ByValue => {
                    this.emit_enum_variant("ByValue", 1, 0, |_| Ok(()))
                }
                ty::UpvarCapture::ByRef(ty::UpvarBorrow { kind, region }) => {
                    this.emit_enum_variant("ByRef", 2, 0, |this| {
                        this.emit_enum_variant_arg(0, |this| kind.encode(this));
                        this.emit_enum_variant_arg(1, |this| region.encode(this))
                    })
                }
            }
        }).unwrap()
    }

    fn emit_auto_adjustment(&mut self, adj: &adjustment::AutoAdjustment<'tcx>) {
        use rustc_serialize::Encoder;

        self.emit_enum("AutoAdjustment", |this| {
            match *adj {
                adjustment::AdjustReifyFnPointer => {
                    this.emit_enum_variant("AdjustReifyFnPointer", 1, 0, |_| Ok(()))
                }

                adjustment::AdjustUnsafeFnPointer => {
                    this.emit_enum_variant("AdjustUnsafeFnPointer", 2, 0, |_| {
                        Ok(())
                    })
                }

                adjustment::AdjustMutToConstPointer => {
                    this.emit_enum_variant("AdjustMutToConstPointer", 3, 0, |_| {
                        Ok(())
                    })
                }

                adjustment::AdjustDerefRef(ref auto_deref_ref) => {
                    this.emit_enum_variant("AdjustDerefRef", 4, 2, |this| {
                        this.emit_enum_variant_arg(0,
                            |this| Ok(this.emit_auto_deref_ref(auto_deref_ref)))
                    })
                }

                adjustment::AdjustNeverToAny(ref ty) => {
                    this.emit_enum_variant("AdjustNeverToAny", 5, 1, |this| {
                        this.emit_enum_variant_arg(0, |this| ty.encode(this))
                    })
                }
            }
        });
    }

    fn emit_autoref(&mut self, autoref: &adjustment::AutoRef<'tcx>) {
        use rustc_serialize::Encoder;

        self.emit_enum("AutoRef", |this| {
            match autoref {
                &adjustment::AutoPtr(r, m) => {
                    this.emit_enum_variant("AutoPtr", 0, 2, |this| {
                        this.emit_enum_variant_arg(0, |this| r.encode(this));
                        this.emit_enum_variant_arg(1, |this| m.encode(this))
                    })
                }
                &adjustment::AutoUnsafe(m) => {
                    this.emit_enum_variant("AutoUnsafe", 1, 1, |this| {
                        this.emit_enum_variant_arg(0, |this| m.encode(this))
                    })
                }
            }
        });
    }

    fn emit_auto_deref_ref(&mut self, auto_deref_ref: &adjustment::AutoDerefRef<'tcx>) {
        use rustc_serialize::Encoder;

        self.emit_struct("AutoDerefRef", 2, |this| {
            this.emit_struct_field("autoderefs", 0, |this| auto_deref_ref.autoderefs.encode(this));

            this.emit_struct_field("autoref", 1, |this| {
                this.emit_option(|this| {
                    match auto_deref_ref.autoref {
                        None => this.emit_option_none(),
                        Some(ref a) => this.emit_option_some(|this| Ok(this.emit_autoref(a))),
                    }
                })
            });

            this.emit_struct_field("unsize", 2, |this| {
                auto_deref_ref.unsize.encode(this)
            })
        });
    }

    fn tag<F>(&mut self,
              tag_id: usize,
              f: F) where
        F: FnOnce(&mut Self),
    {
        self.start_tag(tag_id);
        f(self);
        self.end_tag();
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

fn encode_side_tables_for_ii(ecx: &mut EncodeContext, ii: &InlinedItem) {
    ecx.start_tag(c::tag_table);
    ii.visit(&mut SideTableEncodingIdVisitor {
        ecx: ecx
    });
    ecx.end_tag();
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
            ty.encode(ecx);
        })
    }

    if let Some(item_substs) = tcx.tables.borrow().item_substs.get(&id) {
        ecx.tag(c::tag_table_item_subst, |ecx| {
            ecx.id(id);
            item_substs.substs.encode(ecx);
        })
    }

    if let Some(fv) = tcx.freevars.borrow().get(&id) {
        ecx.tag(c::tag_table_freevars, |ecx| {
            ecx.id(id);
            ecx.emit_from_vec(fv, |ecx, fv_entry| {
                fv_entry.encode(ecx)
            });
        });

        for freevar in fv {
            ecx.tag(c::tag_table_upvar_capture_map, |ecx| {
                ecx.id(id);

                let var_id = freevar.def.var_id();
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
                var_id.encode(ecx);
                ecx.emit_upvar_capture(&upvar_capture);
            })
        }
    }

    let method_call = ty::MethodCall::expr(id);
    if let Some(method) = tcx.tables.borrow().method_map.get(&method_call) {
        ecx.tag(c::tag_table_method_map, |ecx| {
            ecx.id(id);
            ecx.encode_method_callee(method_call.autoderef, method)
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
                            ecx.encode_method_callee(method_call.autoderef, method)
                        })
                    }
                }
            }
            _ => {}
        }

        ecx.tag(c::tag_table_adjustments, |ecx| {
            ecx.id(id);
            ecx.emit_auto_adjustment(adjustment);
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

impl<'a, 'tcx> DecodeContext<'a, 'tcx> {
    fn read_upvar_capture(&mut self) -> ty::UpvarCapture<'tcx> {
        self.read_enum("UpvarCapture", |this| {
            let variants = ["ByValue", "ByRef"];
            this.read_enum_variant(&variants, |this, i| {
                Ok(match i {
                    1 => ty::UpvarCapture::ByValue,
                    2 => ty::UpvarCapture::ByRef(ty::UpvarBorrow {
                        kind: this.read_enum_variant_arg(0,
                                  |this| Decodable::decode(this)).unwrap(),
                        region: this.read_enum_variant_arg(1,
                                    |this| Decodable::decode(this)).unwrap()
                    }),
                    _ => bug!("bad enum variant for ty::UpvarCapture")
                })
            })
        }).unwrap()
    }
    fn read_auto_adjustment(&mut self) -> adjustment::AutoAdjustment<'tcx> {
        self.read_enum("AutoAdjustment", |this| {
            let variants = ["AdjustReifyFnPointer", "AdjustUnsafeFnPointer",
                            "AdjustMutToConstPointer", "AdjustDerefRef",
                            "AdjustNeverToAny"];
            this.read_enum_variant(&variants, |this, i| {
                Ok(match i {
                    1 => adjustment::AdjustReifyFnPointer,
                    2 => adjustment::AdjustUnsafeFnPointer,
                    3 => adjustment::AdjustMutToConstPointer,
                    4 => {
                        let auto_deref_ref: adjustment::AutoDerefRef =
                            this.read_enum_variant_arg(0,
                                |this| Ok(this.read_auto_deref_ref())).unwrap();

                        adjustment::AdjustDerefRef(auto_deref_ref)
                    }
                    5 => {
                        let ty: Ty<'tcx> = this.read_enum_variant_arg(0, |this| {
                            Ty::decode(this)
                        }).unwrap();

                        adjustment::AdjustNeverToAny(ty)
                    }
                    _ => bug!("bad enum variant for adjustment::AutoAdjustment")
                })
            })
        }).unwrap()
    }

    fn read_auto_deref_ref(&mut self) -> adjustment::AutoDerefRef<'tcx> {
        self.read_struct("AutoDerefRef", 2, |this| {
            Ok(adjustment::AutoDerefRef {
                autoderefs: this.read_struct_field("autoderefs", 0, |this| {
                    Decodable::decode(this)
                }).unwrap(),
                autoref: this.read_struct_field("autoref", 1, |this| {
                    this.read_option(|this, b| {
                        if b {
                            Ok(Some(this.read_autoref()))
                        } else {
                            Ok(None)
                        }
                    })
                }).unwrap(),
                unsize: this.read_struct_field("unsize", 2, |this| {
                    Decodable::decode(this)
                }).unwrap(),
            })
        }).unwrap()
    }

    fn read_autoref(&mut self) -> adjustment::AutoRef<'tcx> {
        self.read_enum("AutoRef", |this| {
            let variants = ["AutoPtr", "AutoUnsafe"];
            this.read_enum_variant(&variants, |this, i| {
                Ok(match i {
                    0 => {
                        let r: &'tcx ty::Region =
                            this.read_enum_variant_arg(0, |this| {
                                Decodable::decode(this)
                            }).unwrap();
                        let m: hir::Mutability =
                            this.read_enum_variant_arg(1, |this| {
                                Decodable::decode(this)
                            }).unwrap();

                        adjustment::AutoPtr(r, m)
                    }
                    1 => {
                        let m: hir::Mutability =
                            this.read_enum_variant_arg(0, |this| Decodable::decode(this)).unwrap();

                        adjustment::AutoUnsafe(m)
                    }
                    _ => bug!("bad enum variant for adjustment::AutoRef")
                })
            })
        }).unwrap()
    }
}

fn decode_side_tables<'a, 'tcx>(dcx: &mut DecodeContext<'a, 'tcx>,
                                ast_doc: rbml::Doc<'a>) {
    for (tag, entry_doc) in reader::docs(ast_doc.get(c::tag_table)) {
        dcx.rbml_r = reader::Decoder::new(entry_doc);

        let id0: ast::NodeId = Decodable::decode(dcx).unwrap();
        let id = dcx.tr_id(id0);

        debug!(">> Side table document with tag 0x{:x} \
                found for id {} (orig {})",
               tag, id, id0);

        match tag {
            c::tag_table_def => {
                let def = Def::decode(dcx).unwrap().tr(dcx);
                dcx.tcx.def_map.borrow_mut().insert(id, def::PathResolution::new(def));
            }
            c::tag_table_node_type => {
                let ty = Ty::decode(dcx).unwrap();
                dcx.tcx.node_type_insert(id, ty);
            }
            c::tag_table_item_subst => {
                let item_substs = ty::ItemSubsts {
                    substs: Decodable::decode(dcx).unwrap()
                };
                dcx.tcx.tables.borrow_mut().item_substs.insert(
                    id, item_substs);
            }
            c::tag_table_freevars => {
                let fv_info = dcx.read_to_vec(|dcx| {
                    Ok(dcx.read_freevar_entry())
                }).unwrap().into_iter().collect();
                dcx.tcx.freevars.borrow_mut().insert(id, fv_info);
            }
            c::tag_table_upvar_capture_map => {
                let var_id = ast::NodeId::decode(dcx).unwrap();
                let upvar_id = ty::UpvarId {
                    var_id: dcx.tr_id(var_id),
                    closure_expr_id: id
                };
                let ub = dcx.read_upvar_capture();
                dcx.tcx.tables.borrow_mut().upvar_capture_map.insert(upvar_id, ub);
            }
            c::tag_table_method_map => {
                let (autoderef, method) = dcx.read_method_callee();
                let method_call = ty::MethodCall {
                    expr_id: id,
                    autoderef: autoderef
                };
                dcx.tcx.tables.borrow_mut().method_map.insert(method_call, method);
            }
            c::tag_table_adjustments => {
                let adj = dcx.read_auto_adjustment();
                dcx.tcx.tables.borrow_mut().adjustments.insert(id, adj);
            }
            c::tag_table_cast_kinds => {
                let cast_kind = cast::CastKind::decode(dcx).unwrap();
                dcx.tcx.cast_kinds.borrow_mut().insert(id, cast_kind);
            }
            c::tag_table_const_qualif => {
                let qualif = ConstQualif::decode(dcx).unwrap();
                dcx.tcx.const_qualif_map.borrow_mut().insert(id, qualif);
            }
            _ => {
                bug!("unknown tag found in side tables: {:x}", tag);
            }
        }

        debug!(">< Side table doc loaded");
    }
}

fn inlined_item_id_range(ii: &InlinedItem) -> IdRange {
    let mut visitor = IdRangeComputingVisitor::new();
    ii.visit(&mut visitor);
    visitor.result()
}
