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

use rustc::front::map as ast_map;
use rustc::session::Session;

use rustc_front::hir;
use rustc_front::fold;
use rustc_front::fold::Folder;

use common as c;
use cstore;
use decoder;
use encoder as e;
use tydecode;
use tyencode;

use middle::cstore::{InlinedItem, InlinedItemRef};
use middle::ty::adjustment;
use middle::ty::cast;
use middle::check_const::ConstQualif;
use middle::def;
use middle::def_id::DefId;
use middle::privacy::{AllPublic, LastMod};
use middle::region;
use middle::subst;
use middle::ty::{self, Ty};

use syntax::{ast, ast_util, codemap};
use syntax::ast::NodeIdAssigner;
use syntax::ptr::P;

use std::cell::Cell;
use std::io::SeekFrom;
use std::io::prelude::*;
use std::fmt::Debug;

use rbml::reader;
use rbml::writer::Encoder;
use rbml;
use serialize;
use serialize::{Decodable, Decoder, DecoderHelpers, Encodable};
use serialize::EncoderHelpers;

#[cfg(test)] use std::io::Cursor;
#[cfg(test)] use syntax::parse;
#[cfg(test)] use syntax::ast::NodeId;
#[cfg(test)] use rustc_front::print::pprust;
#[cfg(test)] use rustc_front::lowering::{lower_item, LoweringContext};

struct DecodeContext<'a, 'b, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    cdata: &'b cstore::crate_metadata,
    from_id_range: ast_util::IdRange,
    to_id_range: ast_util::IdRange,
    // Cache the last used filemap for translating spans as an optimization.
    last_filemap_index: Cell<usize>,
}

trait tr {
    fn tr(&self, dcx: &DecodeContext) -> Self;
}

// ______________________________________________________________________
// Top-level methods.

pub fn encode_inlined_item(ecx: &e::EncodeContext,
                           rbml_w: &mut Encoder,
                           ii: InlinedItemRef) {
    let id = match ii {
        InlinedItemRef::Item(i) => i.id,
        InlinedItemRef::Foreign(i) => i.id,
        InlinedItemRef::TraitItem(_, ti) => ti.id,
        InlinedItemRef::ImplItem(_, ii) => ii.id,
    };
    debug!("> Encoding inlined item: {} ({:?})",
           ecx.tcx.map.path_to_string(id),
           rbml_w.writer.seek(SeekFrom::Current(0)));

    // Folding could be avoided with a smarter encoder.
    let ii = simplify_ast(ii);
    let id_range = inlined_item_id_range(&ii);

    rbml_w.start_tag(c::tag_ast as usize);
    id_range.encode(rbml_w);
    encode_ast(rbml_w, &ii);
    encode_side_tables_for_ii(ecx, rbml_w, &ii);
    rbml_w.end_tag();

    debug!("< Encoded inlined fn: {} ({:?})",
           ecx.tcx.map.path_to_string(id),
           rbml_w.writer.seek(SeekFrom::Current(0)));
}

impl<'a, 'b, 'c, 'tcx> ast_map::FoldOps for &'a DecodeContext<'b, 'c, 'tcx> {
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
    fn new_span(&self, span: codemap::Span) -> codemap::Span {
        self.tr_span(span)
    }
}

/// Decodes an item from its AST in the cdata's metadata and adds it to the
/// ast-map.
pub fn decode_inlined_item<'tcx>(cdata: &cstore::crate_metadata,
                                 tcx: &ty::ctxt<'tcx>,
                                 path: Vec<ast_map::PathElem>,
                                 def_path: ast_map::DefPath,
                                 par_doc: rbml::Doc,
                                 orig_did: DefId)
                                 -> Result<&'tcx InlinedItem, (Vec<ast_map::PathElem>,
                                                               ast_map::DefPath)> {
    match par_doc.opt_child(c::tag_ast) {
      None => Err((path, def_path)),
      Some(ast_doc) => {
        let mut path_as_str = None;
        debug!("> Decoding inlined fn: {:?}::?",
        {
            // Do an Option dance to use the path after it is moved below.
            let s = ast_map::path_to_string(path.iter().cloned());
            path_as_str = Some(s);
            path_as_str.as_ref().map(|x| &x[..])
        });
        let mut ast_dsr = reader::Decoder::new(ast_doc);
        let from_id_range = Decodable::decode(&mut ast_dsr).unwrap();
        let to_id_range = reserve_id_range(&tcx.sess, from_id_range);
        let dcx = &DecodeContext {
            cdata: cdata,
            tcx: tcx,
            from_id_range: from_id_range,
            to_id_range: to_id_range,
            last_filemap_index: Cell::new(0)
        };
        let raw_ii = decode_ast(ast_doc);
        let ii = ast_map::map_decoded_item(&dcx.tcx.map, path, def_path, raw_ii, dcx);

        let name = match *ii {
            InlinedItem::Item(ref i) => i.name,
            InlinedItem::Foreign(ref i) => i.name,
            InlinedItem::TraitItem(_, ref ti) => ti.name,
            InlinedItem::ImplItem(_, ref ii) => ii.name
        };
        debug!("Fn named: {}", name);
        debug!("< Decoded inlined fn: {}::{}",
               path_as_str.unwrap(),
               name);
        region::resolve_inlined_item(&tcx.sess, &tcx.region_maps, ii);
        decode_side_tables(dcx, ast_doc);
        copy_item_types(dcx, ii, orig_did);
        match *ii {
          InlinedItem::Item(ref i) => {
            debug!(">>> DECODED ITEM >>>\n{}\n<<< DECODED ITEM <<<",
                   ::rustc_front::print::pprust::item_to_string(&**i));
          }
          _ => { }
        }
        Ok(ii)
      }
    }
}

// ______________________________________________________________________
// Enumerating the IDs which appear in an AST

fn reserve_id_range(sess: &Session,
                    from_id_range: ast_util::IdRange) -> ast_util::IdRange {
    // Handle the case of an empty range:
    if from_id_range.empty() { return from_id_range; }
    let cnt = from_id_range.max - from_id_range.min;
    let to_id_min = sess.reserve_node_ids(cnt);
    let to_id_max = to_id_min + cnt;
    ast_util::IdRange { min: to_id_min, max: to_id_max }
}

impl<'a, 'b, 'tcx> DecodeContext<'a, 'b, 'tcx> {
    /// Translates an internal id, meaning a node id that is known to refer to some part of the
    /// item currently being inlined, such as a local variable or argument.  All naked node-ids
    /// that appear in types have this property, since if something might refer to an external item
    /// we would use a def-id to allow for the possibility that the item resides in another crate.
    pub fn tr_id(&self, id: ast::NodeId) -> ast::NodeId {
        // from_id_range should be non-empty
        assert!(!self.from_id_range.empty());
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
    pub fn tr_span(&self, span: codemap::Span) -> codemap::Span {
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

impl tr for codemap::Span {
    fn tr(&self, dcx: &DecodeContext) -> codemap::Span {
        dcx.tr_span(*self)
    }
}

trait def_id_encoder_helpers {
    fn emit_def_id(&mut self, did: DefId);
}

impl<S:serialize::Encoder> def_id_encoder_helpers for S
    where <S as serialize::serialize::Encoder>::Error: Debug
{
    fn emit_def_id(&mut self, did: DefId) {
        did.encode(self).unwrap()
    }
}

trait def_id_decoder_helpers {
    fn read_def_id(&mut self, dcx: &DecodeContext) -> DefId;
    fn read_def_id_nodcx(&mut self,
                         cdata: &cstore::crate_metadata) -> DefId;
}

impl<D:serialize::Decoder> def_id_decoder_helpers for D
    where <D as serialize::serialize::Decoder>::Error: Debug
{
    fn read_def_id(&mut self, dcx: &DecodeContext) -> DefId {
        let did: DefId = Decodable::decode(self).unwrap();
        did.tr(dcx)
    }

    fn read_def_id_nodcx(&mut self,
                         cdata: &cstore::crate_metadata)
                         -> DefId {
        let did: DefId = Decodable::decode(self).unwrap();
        decoder::translate_def_id(cdata, did)
    }
}

// ______________________________________________________________________
// Encoding and decoding the AST itself
//
// When decoding, we have to renumber the AST so that the node ids that
// appear within are disjoint from the node ids in our existing ASTs.
// We also have to adjust the spans: for now we just insert a dummy span,
// but eventually we should add entries to the local codemap as required.

fn encode_ast(rbml_w: &mut Encoder, item: &InlinedItem) {
    rbml_w.start_tag(c::tag_tree as usize);
    item.encode(rbml_w);
    rbml_w.end_tag();
}

struct NestedItemsDropper;

impl Folder for NestedItemsDropper {
    fn fold_block(&mut self, blk: P<hir::Block>) -> P<hir::Block> {
        blk.and_then(|hir::Block {id, stmts, expr, rules, span, ..}| {
            let stmts_sans_items = stmts.into_iter().filter_map(|stmt| {
                let use_stmt = match stmt.node {
                    hir::StmtExpr(_, _) | hir::StmtSemi(_, _) => true,
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
fn simplify_ast(ii: InlinedItemRef) -> InlinedItem {
    let mut fld = NestedItemsDropper;

    match ii {
        // HACK we're not dropping items.
        InlinedItemRef::Item(i) => {
            InlinedItem::Item(P(fold::noop_fold_item(i.clone(), &mut fld)))
        }
        InlinedItemRef::TraitItem(d, ti) => {
            InlinedItem::TraitItem(d, P(fold::noop_fold_trait_item(ti.clone(), &mut fld)))
        }
        InlinedItemRef::ImplItem(d, ii) => {
            InlinedItem::ImplItem(d, P(fold::noop_fold_impl_item(ii.clone(), &mut fld)))
        }
        InlinedItemRef::Foreign(i) => {
            InlinedItem::Foreign(P(fold::noop_fold_foreign_item(i.clone(), &mut fld)))
        }
    }
}

fn decode_ast(par_doc: rbml::Doc) -> InlinedItem {
    let chi_doc = par_doc.get(c::tag_tree as usize);
    let mut d = reader::Decoder::new(chi_doc);
    Decodable::decode(&mut d).unwrap()
}

// ______________________________________________________________________
// Encoding and decoding of ast::def

fn decode_def(dcx: &DecodeContext, dsr: &mut reader::Decoder) -> def::Def {
    let def: def::Def = Decodable::decode(dsr).unwrap();
    def.tr(dcx)
}

impl tr for def::Def {
    fn tr(&self, dcx: &DecodeContext) -> def::Def {
        match *self {
          def::DefFn(did, is_ctor) => def::DefFn(did.tr(dcx), is_ctor),
          def::DefMethod(did) => def::DefMethod(did.tr(dcx)),
          def::DefSelfTy(opt_did, impl_ids) => { def::DefSelfTy(opt_did.map(|did| did.tr(dcx)),
                                                                impl_ids.map(|(nid1, nid2)| {
                                                                    (dcx.tr_id(nid1),
                                                                     dcx.tr_id(nid2))
                                                                })) }
          def::DefMod(did) => { def::DefMod(did.tr(dcx)) }
          def::DefForeignMod(did) => { def::DefForeignMod(did.tr(dcx)) }
          def::DefStatic(did, m) => { def::DefStatic(did.tr(dcx), m) }
          def::DefConst(did) => { def::DefConst(did.tr(dcx)) }
          def::DefAssociatedConst(did) => def::DefAssociatedConst(did.tr(dcx)),
          def::DefLocal(_, nid) => {
              let nid = dcx.tr_id(nid);
              let did = dcx.tcx.map.local_def_id(nid);
              def::DefLocal(did, nid)
          }
          def::DefVariant(e_did, v_did, is_s) => {
            def::DefVariant(e_did.tr(dcx), v_did.tr(dcx), is_s)
          },
          def::DefTrait(did) => def::DefTrait(did.tr(dcx)),
          def::DefTy(did, is_enum) => def::DefTy(did.tr(dcx), is_enum),
          def::DefAssociatedTy(trait_did, did) =>
              def::DefAssociatedTy(trait_did.tr(dcx), did.tr(dcx)),
          def::DefPrimTy(p) => def::DefPrimTy(p),
          def::DefTyParam(s, index, def_id, n) => def::DefTyParam(s, index, def_id.tr(dcx), n),
          def::DefUse(did) => def::DefUse(did.tr(dcx)),
          def::DefUpvar(_, nid1, index, nid2) => {
              let nid1 = dcx.tr_id(nid1);
              let nid2 = dcx.tr_id(nid2);
              let did1 = dcx.tcx.map.local_def_id(nid1);
              def::DefUpvar(did1, nid1, index, nid2)
          }
          def::DefStruct(did) => def::DefStruct(did.tr(dcx)),
          def::DefLabel(nid) => def::DefLabel(dcx.tr_id(nid)),
          def::DefErr => def::DefErr,
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of freevar information

fn encode_freevar_entry(rbml_w: &mut Encoder, fv: &ty::Freevar) {
    (*fv).encode(rbml_w).unwrap();
}

trait rbml_decoder_helper {
    fn read_freevar_entry(&mut self, dcx: &DecodeContext)
                          -> ty::Freevar;
    fn read_capture_mode(&mut self) -> hir::CaptureClause;
}

impl<'a> rbml_decoder_helper for reader::Decoder<'a> {
    fn read_freevar_entry(&mut self, dcx: &DecodeContext)
                          -> ty::Freevar {
        let fv: ty::Freevar = Decodable::decode(self).unwrap();
        fv.tr(dcx)
    }

    fn read_capture_mode(&mut self) -> hir::CaptureClause {
        let cm: hir::CaptureClause = Decodable::decode(self).unwrap();
        cm
    }
}

impl tr for ty::Freevar {
    fn tr(&self, dcx: &DecodeContext) -> ty::Freevar {
        ty::Freevar {
            def: self.def.tr(dcx),
            span: self.span.tr(dcx),
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of MethodCallee

trait read_method_callee_helper<'tcx> {
    fn read_method_callee<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                  -> (u32, ty::MethodCallee<'tcx>);
}

fn encode_method_callee<'a, 'tcx>(ecx: &e::EncodeContext<'a, 'tcx>,
                                  rbml_w: &mut Encoder,
                                  autoderef: u32,
                                  method: &ty::MethodCallee<'tcx>) {
    use serialize::Encoder;

    rbml_w.emit_struct("MethodCallee", 4, |rbml_w| {
        rbml_w.emit_struct_field("autoderef", 0, |rbml_w| {
            autoderef.encode(rbml_w)
        });
        rbml_w.emit_struct_field("def_id", 1, |rbml_w| {
            Ok(rbml_w.emit_def_id(method.def_id))
        });
        rbml_w.emit_struct_field("ty", 2, |rbml_w| {
            Ok(rbml_w.emit_ty(ecx, method.ty))
        });
        rbml_w.emit_struct_field("substs", 3, |rbml_w| {
            Ok(rbml_w.emit_substs(ecx, &method.substs))
        })
    }).unwrap();
}

impl<'a, 'tcx> read_method_callee_helper<'tcx> for reader::Decoder<'a> {
    fn read_method_callee<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                                  -> (u32, ty::MethodCallee<'tcx>) {

        self.read_struct("MethodCallee", 4, |this| {
            let autoderef = this.read_struct_field("autoderef", 0,
                                                   Decodable::decode).unwrap();
            Ok((autoderef, ty::MethodCallee {
                def_id: this.read_struct_field("def_id", 1, |this| {
                    Ok(this.read_def_id(dcx))
                }).unwrap(),
                ty: this.read_struct_field("ty", 2, |this| {
                    Ok(this.read_ty(dcx))
                }).unwrap(),
                substs: this.read_struct_field("substs", 3, |this| {
                    Ok(dcx.tcx.mk_substs(this.read_substs(dcx)))
                }).unwrap()
            }))
        }).unwrap()
    }
}

pub fn encode_cast_kind(ebml_w: &mut Encoder, kind: cast::CastKind) {
    kind.encode(ebml_w).unwrap();
}

// ______________________________________________________________________
// Encoding and decoding the side tables

trait get_ty_str_ctxt<'tcx> {
    fn ty_str_ctxt<'a>(&'a self) -> tyencode::ctxt<'a, 'tcx>;
}

impl<'a, 'tcx> get_ty_str_ctxt<'tcx> for e::EncodeContext<'a, 'tcx> {
    fn ty_str_ctxt<'b>(&'b self) -> tyencode::ctxt<'b, 'tcx> {
        tyencode::ctxt {
            diag: self.tcx.sess.diagnostic(),
            ds: e::def_to_string,
            tcx: self.tcx,
            abbrevs: &self.type_abbrevs
        }
    }
}

trait rbml_writer_helpers<'tcx> {
    fn emit_region(&mut self, ecx: &e::EncodeContext, r: ty::Region);
    fn emit_ty<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>, ty: Ty<'tcx>);
    fn emit_tys<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>, tys: &[Ty<'tcx>]);
    fn emit_predicate<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                          predicate: &ty::Predicate<'tcx>);
    fn emit_trait_ref<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                          ty: &ty::TraitRef<'tcx>);
    fn emit_substs<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                       substs: &subst::Substs<'tcx>);
    fn emit_existential_bounds<'b>(&mut self, ecx: &e::EncodeContext<'b,'tcx>,
                                   bounds: &ty::ExistentialBounds<'tcx>);
    fn emit_builtin_bounds(&mut self, ecx: &e::EncodeContext, bounds: &ty::BuiltinBounds);
    fn emit_upvar_capture(&mut self, ecx: &e::EncodeContext, capture: &ty::UpvarCapture);
    fn emit_auto_adjustment<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                                adj: &adjustment::AutoAdjustment<'tcx>);
    fn emit_autoref<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                        autoref: &adjustment::AutoRef<'tcx>);
    fn emit_auto_deref_ref<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                               auto_deref_ref: &adjustment::AutoDerefRef<'tcx>);
}

impl<'a, 'tcx> rbml_writer_helpers<'tcx> for Encoder<'a> {
    fn emit_region(&mut self, ecx: &e::EncodeContext, r: ty::Region) {
        self.emit_opaque(|this| Ok(e::write_region(ecx, this, r)));
    }

    fn emit_ty<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>, ty: Ty<'tcx>) {
        self.emit_opaque(|this| Ok(e::write_type(ecx, this, ty)));
    }

    fn emit_tys<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>, tys: &[Ty<'tcx>]) {
        self.emit_from_vec(tys, |this, ty| Ok(this.emit_ty(ecx, *ty)));
    }

    fn emit_trait_ref<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>,
                          trait_ref: &ty::TraitRef<'tcx>) {
        self.emit_opaque(|this| Ok(e::write_trait_ref(ecx, this, trait_ref)));
    }

    fn emit_predicate<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>,
                          predicate: &ty::Predicate<'tcx>) {
        self.emit_opaque(|this| {
            Ok(tyencode::enc_predicate(this,
                                       &ecx.ty_str_ctxt(),
                                       predicate))
        });
    }

    fn emit_existential_bounds<'b>(&mut self, ecx: &e::EncodeContext<'b,'tcx>,
                                   bounds: &ty::ExistentialBounds<'tcx>) {
        self.emit_opaque(|this| Ok(tyencode::enc_existential_bounds(this,
                                                                    &ecx.ty_str_ctxt(),
                                                                    bounds)));
    }

    fn emit_builtin_bounds(&mut self, ecx: &e::EncodeContext, bounds: &ty::BuiltinBounds) {
        self.emit_opaque(|this| Ok(tyencode::enc_builtin_bounds(this,
                                                                &ecx.ty_str_ctxt(),
                                                                bounds)));
    }

    fn emit_upvar_capture(&mut self, ecx: &e::EncodeContext, capture: &ty::UpvarCapture) {
        use serialize::Encoder;

        self.emit_enum("UpvarCapture", |this| {
            match *capture {
                ty::UpvarCapture::ByValue => {
                    this.emit_enum_variant("ByValue", 1, 0, |_| Ok(()))
                }
                ty::UpvarCapture::ByRef(ty::UpvarBorrow { kind, region }) => {
                    this.emit_enum_variant("ByRef", 2, 0, |this| {
                        this.emit_enum_variant_arg(0,
                            |this| kind.encode(this));
                        this.emit_enum_variant_arg(1,
                            |this| Ok(this.emit_region(ecx, region)))
                    })
                }
            }
        }).unwrap()
    }

    fn emit_substs<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>,
                       substs: &subst::Substs<'tcx>) {
        self.emit_opaque(|this| Ok(tyencode::enc_substs(this,
                                                           &ecx.ty_str_ctxt(),
                                                           substs)));
    }

    fn emit_auto_adjustment<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>,
                                adj: &adjustment::AutoAdjustment<'tcx>) {
        use serialize::Encoder;

        self.emit_enum("AutoAdjustment", |this| {
            match *adj {
                adjustment::AdjustReifyFnPointer=> {
                    this.emit_enum_variant("AdjustReifyFnPointer", 1, 0, |_| Ok(()))
                }

                adjustment::AdjustUnsafeFnPointer => {
                    this.emit_enum_variant("AdjustUnsafeFnPointer", 2, 0, |_| {
                        Ok(())
                    })
                }

                adjustment::AdjustDerefRef(ref auto_deref_ref) => {
                    this.emit_enum_variant("AdjustDerefRef", 3, 2, |this| {
                        this.emit_enum_variant_arg(0,
                            |this| Ok(this.emit_auto_deref_ref(ecx, auto_deref_ref)))
                    })
                }
            }
        });
    }

    fn emit_autoref<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>,
                        autoref: &adjustment::AutoRef<'tcx>) {
        use serialize::Encoder;

        self.emit_enum("AutoRef", |this| {
            match autoref {
                &adjustment::AutoPtr(r, m) => {
                    this.emit_enum_variant("AutoPtr", 0, 2, |this| {
                        this.emit_enum_variant_arg(0,
                            |this| Ok(this.emit_region(ecx, *r)));
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

    fn emit_auto_deref_ref<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>,
                               auto_deref_ref: &adjustment::AutoDerefRef<'tcx>) {
        use serialize::Encoder;

        self.emit_struct("AutoDerefRef", 2, |this| {
            this.emit_struct_field("autoderefs", 0, |this| auto_deref_ref.autoderefs.encode(this));

            this.emit_struct_field("autoref", 1, |this| {
                this.emit_option(|this| {
                    match auto_deref_ref.autoref {
                        None => this.emit_option_none(),
                        Some(ref a) => this.emit_option_some(|this| Ok(this.emit_autoref(ecx, a))),
                    }
                })
            });

            this.emit_struct_field("unsize", 2, |this| {
                this.emit_option(|this| {
                    match auto_deref_ref.unsize {
                        None => this.emit_option_none(),
                        Some(target) => this.emit_option_some(|this| {
                            Ok(this.emit_ty(ecx, target))
                        })
                    }
                })
            })
        });
    }
}

trait write_tag_and_id {
    fn tag<F>(&mut self, tag_id: c::astencode_tag, f: F) where F: FnOnce(&mut Self);
    fn id(&mut self, id: ast::NodeId);
}

impl<'a> write_tag_and_id for Encoder<'a> {
    fn tag<F>(&mut self,
              tag_id: c::astencode_tag,
              f: F) where
        F: FnOnce(&mut Encoder<'a>),
    {
        self.start_tag(tag_id as usize);
        f(self);
        self.end_tag();
    }

    fn id(&mut self, id: ast::NodeId) {
        id.encode(self).unwrap();
    }
}

struct SideTableEncodingIdVisitor<'a, 'b:'a, 'c:'a, 'tcx:'c> {
    ecx: &'a e::EncodeContext<'c, 'tcx>,
    rbml_w: &'a mut Encoder<'b>,
}

impl<'a, 'b, 'c, 'tcx> ast_util::IdVisitingOperation for
        SideTableEncodingIdVisitor<'a, 'b, 'c, 'tcx> {
    fn visit_id(&mut self, id: ast::NodeId) {
        encode_side_tables_for_id(self.ecx, self.rbml_w, id)
    }
}

fn encode_side_tables_for_ii(ecx: &e::EncodeContext,
                             rbml_w: &mut Encoder,
                             ii: &InlinedItem) {
    rbml_w.start_tag(c::tag_table as usize);
    ii.visit_ids(&mut SideTableEncodingIdVisitor {
        ecx: ecx,
        rbml_w: rbml_w
    });
    rbml_w.end_tag();
}

fn encode_side_tables_for_id(ecx: &e::EncodeContext,
                             rbml_w: &mut Encoder,
                             id: ast::NodeId) {
    let tcx = ecx.tcx;

    debug!("Encoding side tables for id {}", id);

    if let Some(def) = tcx.def_map.borrow().get(&id).map(|d| d.full_def()) {
        rbml_w.tag(c::tag_table_def, |rbml_w| {
            rbml_w.id(id);
            def.encode(rbml_w).unwrap();
        })
    }

    if let Some(ty) = tcx.node_types().get(&id) {
        rbml_w.tag(c::tag_table_node_type, |rbml_w| {
            rbml_w.id(id);
            rbml_w.emit_ty(ecx, *ty);
        })
    }

    if let Some(item_substs) = tcx.tables.borrow().item_substs.get(&id) {
        rbml_w.tag(c::tag_table_item_subst, |rbml_w| {
            rbml_w.id(id);
            rbml_w.emit_substs(ecx, &item_substs.substs);
        })
    }

    if let Some(fv) = tcx.freevars.borrow().get(&id) {
        rbml_w.tag(c::tag_table_freevars, |rbml_w| {
            rbml_w.id(id);
            rbml_w.emit_from_vec(fv, |rbml_w, fv_entry| {
                Ok(encode_freevar_entry(rbml_w, fv_entry))
            });
        });

        for freevar in fv {
            rbml_w.tag(c::tag_table_upvar_capture_map, |rbml_w| {
                rbml_w.id(id);

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
                var_id.encode(rbml_w);
                rbml_w.emit_upvar_capture(ecx, &upvar_capture);
            })
        }
    }

    let method_call = ty::MethodCall::expr(id);
    if let Some(method) = tcx.tables.borrow().method_map.get(&method_call) {
        rbml_w.tag(c::tag_table_method_map, |rbml_w| {
            rbml_w.id(id);
            encode_method_callee(ecx, rbml_w, method_call.autoderef, method)
        })
    }

    if let Some(adjustment) = tcx.tables.borrow().adjustments.get(&id) {
        match *adjustment {
            adjustment::AdjustDerefRef(ref adj) => {
                for autoderef in 0..adj.autoderefs {
                    let method_call = ty::MethodCall::autoderef(id, autoderef as u32);
                    if let Some(method) = tcx.tables.borrow().method_map.get(&method_call) {
                        rbml_w.tag(c::tag_table_method_map, |rbml_w| {
                            rbml_w.id(id);
                            encode_method_callee(ecx, rbml_w,
                                                 method_call.autoderef, method)
                        })
                    }
                }
            }
            _ => {}
        }

        rbml_w.tag(c::tag_table_adjustments, |rbml_w| {
            rbml_w.id(id);
            rbml_w.emit_auto_adjustment(ecx, adjustment);
        })
    }

    if let Some(cast_kind) = tcx.cast_kinds.borrow().get(&id) {
        rbml_w.tag(c::tag_table_cast_kinds, |rbml_w| {
            rbml_w.id(id);
            encode_cast_kind(rbml_w, *cast_kind)
        })
    }

    if let Some(qualif) = tcx.const_qualif_map.borrow().get(&id) {
        rbml_w.tag(c::tag_table_const_qualif, |rbml_w| {
            rbml_w.id(id);
            qualif.encode(rbml_w).unwrap()
        })
    }
}

trait doc_decoder_helpers: Sized {
    fn as_int(&self) -> isize;
    fn opt_child(&self, tag: c::astencode_tag) -> Option<Self>;
}

impl<'a> doc_decoder_helpers for rbml::Doc<'a> {
    fn as_int(&self) -> isize { reader::doc_as_u64(*self) as isize }
    fn opt_child(&self, tag: c::astencode_tag) -> Option<rbml::Doc<'a>> {
        reader::maybe_get_doc(*self, tag as usize)
    }
}

trait rbml_decoder_decoder_helpers<'tcx> {
    fn read_ty_encoded<'a, 'b, F, R>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>,
                                     f: F) -> R
        where F: for<'x> FnOnce(&mut tydecode::TyDecoder<'x, 'tcx>) -> R;

    fn read_region(&mut self, dcx: &DecodeContext) -> ty::Region;
    fn read_ty<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>) -> Ty<'tcx>;
    fn read_tys<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>) -> Vec<Ty<'tcx>>;
    fn read_trait_ref<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                              -> ty::TraitRef<'tcx>;
    fn read_poly_trait_ref<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                   -> ty::PolyTraitRef<'tcx>;
    fn read_predicate<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                              -> ty::Predicate<'tcx>;
    fn read_existential_bounds<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                       -> ty::ExistentialBounds<'tcx>;
    fn read_substs<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                           -> subst::Substs<'tcx>;
    fn read_upvar_capture(&mut self, dcx: &DecodeContext)
                          -> ty::UpvarCapture;
    fn read_auto_adjustment<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                    -> adjustment::AutoAdjustment<'tcx>;
    fn read_cast_kind<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                 -> cast::CastKind;
    fn read_auto_deref_ref<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                   -> adjustment::AutoDerefRef<'tcx>;
    fn read_autoref<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                            -> adjustment::AutoRef<'tcx>;
    fn convert_def_id(&mut self,
                      dcx: &DecodeContext,
                      did: DefId)
                      -> DefId;

    // Versions of the type reading functions that don't need the full
    // DecodeContext.
    fn read_ty_nodcx(&mut self,
                     tcx: &ty::ctxt<'tcx>, cdata: &cstore::crate_metadata) -> Ty<'tcx>;
    fn read_tys_nodcx(&mut self,
                      tcx: &ty::ctxt<'tcx>,
                      cdata: &cstore::crate_metadata) -> Vec<Ty<'tcx>>;
    fn read_substs_nodcx(&mut self, tcx: &ty::ctxt<'tcx>,
                         cdata: &cstore::crate_metadata)
                         -> subst::Substs<'tcx>;
}

impl<'a, 'tcx> rbml_decoder_decoder_helpers<'tcx> for reader::Decoder<'a> {
    fn read_ty_nodcx(&mut self,
                     tcx: &ty::ctxt<'tcx>,
                     cdata: &cstore::crate_metadata)
                     -> Ty<'tcx> {
        self.read_opaque(|_, doc| {
            Ok(
                tydecode::TyDecoder::with_doc(tcx, cdata.cnum, doc,
                                              &mut |id| decoder::translate_def_id(cdata, id))
                    .parse_ty())
        }).unwrap()
    }

    fn read_tys_nodcx(&mut self,
                      tcx: &ty::ctxt<'tcx>,
                      cdata: &cstore::crate_metadata) -> Vec<Ty<'tcx>> {
        self.read_to_vec(|this| Ok(this.read_ty_nodcx(tcx, cdata)) )
            .unwrap()
            .into_iter()
            .collect()
    }

    fn read_substs_nodcx(&mut self,
                         tcx: &ty::ctxt<'tcx>,
                         cdata: &cstore::crate_metadata)
                         -> subst::Substs<'tcx>
    {
        self.read_opaque(|_, doc| {
            Ok(
                tydecode::TyDecoder::with_doc(tcx, cdata.cnum, doc,
                                              &mut |id| decoder::translate_def_id(cdata, id))
                    .parse_substs())
        }).unwrap()
    }

    fn read_ty_encoded<'b, 'c, F, R>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>, op: F) -> R
        where F: for<'x> FnOnce(&mut tydecode::TyDecoder<'x,'tcx>) -> R
    {
        return self.read_opaque(|this, doc| {
            debug!("read_ty_encoded({})", type_string(doc));
            Ok(op(
                &mut tydecode::TyDecoder::with_doc(
                    dcx.tcx, dcx.cdata.cnum, doc,
                    &mut |a| this.convert_def_id(dcx, a))))
        }).unwrap();

        fn type_string(doc: rbml::Doc) -> String {
            let mut str = String::new();
            for i in doc.start..doc.end {
                str.push(doc.data[i] as char);
            }
            str
        }
    }
    fn read_region(&mut self, dcx: &DecodeContext) -> ty::Region {
        // Note: regions types embed local node ids.  In principle, we
        // should translate these node ids into the new decode
        // context.  However, we do not bother, because region types
        // are not used during trans. This also applies to read_ty.
        return self.read_ty_encoded(dcx, |decoder| decoder.parse_region());
    }
    fn read_ty<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>) -> Ty<'tcx> {
        return self.read_ty_encoded(dcx, |decoder| decoder.parse_ty());
    }

    fn read_tys<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                        -> Vec<Ty<'tcx>> {
        self.read_to_vec(|this| Ok(this.read_ty(dcx))).unwrap().into_iter().collect()
    }

    fn read_trait_ref<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                              -> ty::TraitRef<'tcx> {
        self.read_ty_encoded(dcx, |decoder| decoder.parse_trait_ref())
    }

    fn read_poly_trait_ref<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                                   -> ty::PolyTraitRef<'tcx> {
        ty::Binder(self.read_ty_encoded(dcx, |decoder| decoder.parse_trait_ref()))
    }

    fn read_predicate<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                              -> ty::Predicate<'tcx>
    {
        self.read_ty_encoded(dcx, |decoder| decoder.parse_predicate())
    }

    fn read_existential_bounds<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                                       -> ty::ExistentialBounds<'tcx>
    {
        self.read_ty_encoded(dcx, |decoder| decoder.parse_existential_bounds())
    }

    fn read_substs<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                           -> subst::Substs<'tcx> {
        self.read_opaque(|this, doc| {
            Ok(tydecode::TyDecoder::with_doc(dcx.tcx, dcx.cdata.cnum, doc,
                                             &mut |a| this.convert_def_id(dcx, a))
               .parse_substs())
        }).unwrap()
    }
    fn read_upvar_capture(&mut self, dcx: &DecodeContext) -> ty::UpvarCapture {
        self.read_enum("UpvarCapture", |this| {
            let variants = ["ByValue", "ByRef"];
            this.read_enum_variant(&variants, |this, i| {
                Ok(match i {
                    1 => ty::UpvarCapture::ByValue,
                    2 => ty::UpvarCapture::ByRef(ty::UpvarBorrow {
                        kind: this.read_enum_variant_arg(0,
                                  |this| Decodable::decode(this)).unwrap(),
                        region: this.read_enum_variant_arg(1,
                                    |this| Ok(this.read_region(dcx))).unwrap()
                    }),
                    _ => panic!("bad enum variant for ty::UpvarCapture")
                })
            })
        }).unwrap()
    }
    fn read_auto_adjustment<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                                    -> adjustment::AutoAdjustment<'tcx> {
        self.read_enum("AutoAdjustment", |this| {
            let variants = ["AdjustReifyFnPointer", "AdjustUnsafeFnPointer", "AdjustDerefRef"];
            this.read_enum_variant(&variants, |this, i| {
                Ok(match i {
                    1 => adjustment::AdjustReifyFnPointer,
                    2 => adjustment::AdjustUnsafeFnPointer,
                    3 => {
                        let auto_deref_ref: adjustment::AutoDerefRef =
                            this.read_enum_variant_arg(0,
                                |this| Ok(this.read_auto_deref_ref(dcx))).unwrap();

                        adjustment::AdjustDerefRef(auto_deref_ref)
                    }
                    _ => panic!("bad enum variant for adjustment::AutoAdjustment")
                })
            })
        }).unwrap()
    }

    fn read_auto_deref_ref<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                                   -> adjustment::AutoDerefRef<'tcx> {
        self.read_struct("AutoDerefRef", 2, |this| {
            Ok(adjustment::AutoDerefRef {
                autoderefs: this.read_struct_field("autoderefs", 0, |this| {
                    Decodable::decode(this)
                }).unwrap(),
                autoref: this.read_struct_field("autoref", 1, |this| {
                    this.read_option(|this, b| {
                        if b {
                            Ok(Some(this.read_autoref(dcx)))
                        } else {
                            Ok(None)
                        }
                    })
                }).unwrap(),
                unsize: this.read_struct_field("unsize", 2, |this| {
                    this.read_option(|this, b| {
                        if b {
                            Ok(Some(this.read_ty(dcx)))
                        } else {
                            Ok(None)
                        }
                    })
                }).unwrap(),
            })
        }).unwrap()
    }

    fn read_autoref<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                            -> adjustment::AutoRef<'tcx> {
        self.read_enum("AutoRef", |this| {
            let variants = ["AutoPtr", "AutoUnsafe"];
            this.read_enum_variant(&variants, |this, i| {
                Ok(match i {
                    0 => {
                        let r: ty::Region =
                            this.read_enum_variant_arg(0, |this| {
                                Ok(this.read_region(dcx))
                            }).unwrap();
                        let m: hir::Mutability =
                            this.read_enum_variant_arg(1, |this| {
                                Decodable::decode(this)
                            }).unwrap();

                        adjustment::AutoPtr(dcx.tcx.mk_region(r), m)
                    }
                    1 => {
                        let m: hir::Mutability =
                            this.read_enum_variant_arg(0, |this| Decodable::decode(this)).unwrap();

                        adjustment::AutoUnsafe(m)
                    }
                    _ => panic!("bad enum variant for adjustment::AutoRef")
                })
            })
        }).unwrap()
    }

    fn read_cast_kind<'b, 'c>(&mut self, _dcx: &DecodeContext<'b, 'c, 'tcx>)
                              -> cast::CastKind
    {
        Decodable::decode(self).unwrap()
    }

    /// Converts a def-id that appears in a type.  The correct
    /// translation will depend on what kind of def-id this is.
    /// This is a subtle point: type definitions are not
    /// inlined into the current crate, so if the def-id names
    /// a nominal type or type alias, then it should be
    /// translated to refer to the source crate.
    ///
    /// However, *type parameters* are cloned along with the function
    /// they are attached to.  So we should translate those def-ids
    /// to refer to the new, cloned copy of the type parameter.
    /// We only see references to free type parameters in the body of
    /// an inlined function. In such cases, we need the def-id to
    /// be a local id so that the TypeContents code is able to lookup
    /// the relevant info in the ty_param_defs table.
    ///
    /// *Region parameters*, unfortunately, are another kettle of fish.
    /// In such cases, def_id's can appear in types to distinguish
    /// shadowed bound regions and so forth. It doesn't actually
    /// matter so much what we do to these, since regions are erased
    /// at trans time, but it's good to keep them consistent just in
    /// case. We translate them with `tr_def_id()` which will map
    /// the crate numbers back to the original source crate.
    ///
    /// Scopes will end up as being totally bogus. This can actually
    /// be fixed though.
    ///
    /// Unboxed closures are cloned along with the function being
    /// inlined, and all side tables use interned node IDs, so we
    /// translate their def IDs accordingly.
    ///
    /// It'd be really nice to refactor the type repr to not include
    /// def-ids so that all these distinctions were unnecessary.
    fn convert_def_id(&mut self,
                      dcx: &DecodeContext,
                      did: DefId)
                      -> DefId {
        let r = dcx.tr_def_id(did);
        debug!("convert_def_id(did={:?})={:?}", did, r);
        return r;
    }
}

fn decode_side_tables(dcx: &DecodeContext,
                      ast_doc: rbml::Doc) {
    let tbl_doc = ast_doc.get(c::tag_table as usize);
    for (tag, entry_doc) in reader::docs(tbl_doc) {
        let mut entry_dsr = reader::Decoder::new(entry_doc);
        let id0: ast::NodeId = Decodable::decode(&mut entry_dsr).unwrap();
        let id = dcx.tr_id(id0);

        debug!(">> Side table document with tag 0x{:x} \
                found for id {} (orig {})",
               tag, id, id0);
        let tag = tag as u32;
        let decoded_tag: Option<c::astencode_tag> = c::astencode_tag::from_u32(tag);
        match decoded_tag {
            None => {
                dcx.tcx.sess.bug(
                    &format!("unknown tag found in side tables: {:x}",
                            tag));
            }
            Some(value) => {
                let val_dsr = &mut entry_dsr;

                match value {
                    c::tag_table_def => {
                        let def = decode_def(dcx, val_dsr);
                        dcx.tcx.def_map.borrow_mut().insert(id, def::PathResolution {
                            base_def: def,
                            // This doesn't matter cross-crate.
                            last_private: LastMod(AllPublic),
                            depth: 0
                        });
                    }
                    c::tag_table_node_type => {
                        let ty = val_dsr.read_ty(dcx);
                        debug!("inserting ty for node {}: {:?}",
                               id,  ty);
                        dcx.tcx.node_type_insert(id, ty);
                    }
                    c::tag_table_item_subst => {
                        let item_substs = ty::ItemSubsts {
                            substs: val_dsr.read_substs(dcx)
                        };
                        dcx.tcx.tables.borrow_mut().item_substs.insert(
                            id, item_substs);
                    }
                    c::tag_table_freevars => {
                        let fv_info = val_dsr.read_to_vec(|val_dsr| {
                            Ok(val_dsr.read_freevar_entry(dcx))
                        }).unwrap().into_iter().collect();
                        dcx.tcx.freevars.borrow_mut().insert(id, fv_info);
                    }
                    c::tag_table_upvar_capture_map => {
                        let var_id: ast::NodeId = Decodable::decode(val_dsr).unwrap();
                        let upvar_id = ty::UpvarId {
                            var_id: dcx.tr_id(var_id),
                            closure_expr_id: id
                        };
                        let ub = val_dsr.read_upvar_capture(dcx);
                        dcx.tcx.tables.borrow_mut().upvar_capture_map.insert(upvar_id, ub);
                    }
                    c::tag_table_method_map => {
                        let (autoderef, method) = val_dsr.read_method_callee(dcx);
                        let method_call = ty::MethodCall {
                            expr_id: id,
                            autoderef: autoderef
                        };
                        dcx.tcx.tables.borrow_mut().method_map.insert(method_call, method);
                    }
                    c::tag_table_adjustments => {
                        let adj =
                            val_dsr.read_auto_adjustment(dcx);
                        dcx.tcx.tables.borrow_mut().adjustments.insert(id, adj);
                    }
                    c::tag_table_cast_kinds => {
                        let cast_kind =
                            val_dsr.read_cast_kind(dcx);
                        dcx.tcx.cast_kinds.borrow_mut().insert(id, cast_kind);
                    }
                    c::tag_table_const_qualif => {
                        let qualif: ConstQualif = Decodable::decode(val_dsr).unwrap();
                        dcx.tcx.const_qualif_map.borrow_mut().insert(id, qualif);
                    }
                    _ => {
                        dcx.tcx.sess.bug(
                            &format!("unknown tag found in side tables: {:x}",
                                    tag));
                    }
                }
            }
        }

        debug!(">< Side table doc loaded");
    }
}

// copy the tcache entries from the original item to the new
// inlined item
fn copy_item_types(dcx: &DecodeContext, ii: &InlinedItem, orig_did: DefId) {
    fn copy_item_type(dcx: &DecodeContext,
                      inlined_id: ast::NodeId,
                      remote_did: DefId) {
        let inlined_did = dcx.tcx.map.local_def_id(inlined_id);
        dcx.tcx.register_item_type(inlined_did,
                                   dcx.tcx.lookup_item_type(remote_did));

    }
    // copy the entry for the item itself
    let item_node_id = match ii {
        &InlinedItem::Item(ref i) => i.id,
        &InlinedItem::TraitItem(_, ref ti) => ti.id,
        &InlinedItem::ImplItem(_, ref ii) => ii.id,
        &InlinedItem::Foreign(ref fi) => fi.id
    };
    copy_item_type(dcx, item_node_id, orig_did);

    // copy the entries of inner items
    if let &InlinedItem::Item(ref item) = ii {
        match item.node {
            hir::ItemEnum(ref def, _) => {
                let orig_def = dcx.tcx.lookup_adt_def(orig_did);
                for (i_variant, orig_variant) in
                    def.variants.iter().zip(orig_def.variants.iter())
                {
                    debug!("astencode: copying variant {:?} => {:?}",
                           orig_variant.did, i_variant.node.data.id());
                    copy_item_type(dcx, i_variant.node.data.id(), orig_variant.did);
                }
            }
            hir::ItemStruct(ref def, _) => {
                if !def.is_struct() {
                    let ctor_did = dcx.tcx.lookup_adt_def(orig_did)
                        .struct_variant().did;
                    debug!("astencode: copying ctor {:?} => {:?}", ctor_did,
                           def.id());
                    copy_item_type(dcx, def.id(), ctor_did);
                }
            }
            _ => {}
        }
    }
}

fn inlined_item_id_range(v: &InlinedItem) -> ast_util::IdRange {
    let mut visitor = ast_util::IdRangeComputingVisitor::new();
    v.visit_ids(&mut visitor);
    visitor.result()
}

// ______________________________________________________________________
// Testing of astencode_gen

#[cfg(test)]
fn encode_item_ast(rbml_w: &mut Encoder, item: &hir::Item) {
    rbml_w.start_tag(c::tag_tree as usize);
    (*item).encode(rbml_w);
    rbml_w.end_tag();
}

#[cfg(test)]
fn decode_item_ast(par_doc: rbml::Doc) -> hir::Item {
    let chi_doc = par_doc.get(c::tag_tree as usize);
    let mut d = reader::Decoder::new(chi_doc);
    Decodable::decode(&mut d).unwrap()
}

#[cfg(test)]
trait FakeExtCtxt {
    fn call_site(&self) -> codemap::Span;
    fn cfg(&self) -> ast::CrateConfig;
    fn ident_of(&self, st: &str) -> ast::Ident;
    fn name_of(&self, st: &str) -> ast::Name;
    fn parse_sess(&self) -> &parse::ParseSess;
}

#[cfg(test)]
impl FakeExtCtxt for parse::ParseSess {
    fn call_site(&self) -> codemap::Span {
        codemap::Span {
            lo: codemap::BytePos(0),
            hi: codemap::BytePos(0),
            expn_id: codemap::NO_EXPANSION,
        }
    }
    fn cfg(&self) -> ast::CrateConfig { Vec::new() }
    fn ident_of(&self, st: &str) -> ast::Ident {
        parse::token::str_to_ident(st)
    }
    fn name_of(&self, st: &str) -> ast::Name {
        parse::token::intern(st)
    }
    fn parse_sess(&self) -> &parse::ParseSess { self }
}

#[cfg(test)]
struct FakeNodeIdAssigner;

#[cfg(test)]
// It should go without saying that this may give unexpected results. Avoid
// lowering anything which needs new nodes.
impl NodeIdAssigner for FakeNodeIdAssigner {
    fn next_node_id(&self) -> NodeId {
        0
    }

    fn peek_node_id(&self) -> NodeId {
        0
    }
}

#[cfg(test)]
fn mk_ctxt() -> parse::ParseSess {
    parse::ParseSess::new()
}

#[cfg(test)]
fn roundtrip(in_item: hir::Item) {
    let mut wr = Cursor::new(Vec::new());
    encode_item_ast(&mut Encoder::new(&mut wr), &in_item);
    let rbml_doc = rbml::Doc::new(wr.get_ref());
    let out_item = decode_item_ast(rbml_doc);

    assert!(in_item == out_item);
}

#[test]
fn test_basic() {
    let cx = mk_ctxt();
    let fnia = FakeNodeIdAssigner;
    let lcx = LoweringContext::new(&fnia, None);
    roundtrip(lower_item(&lcx, &quote_item!(&cx,
        fn foo() {}
    ).unwrap()));
}

#[test]
fn test_smalltalk() {
    let cx = mk_ctxt();
    let fnia = FakeNodeIdAssigner;
    let lcx = LoweringContext::new(&fnia, None);
    roundtrip(lower_item(&lcx, &quote_item!(&cx,
        fn foo() -> isize { 3 + 4 } // first smalltalk program ever executed.
    ).unwrap()));
}

#[test]
fn test_more() {
    let cx = mk_ctxt();
    let fnia = FakeNodeIdAssigner;
    let lcx = LoweringContext::new(&fnia, None);
    roundtrip(lower_item(&lcx, &quote_item!(&cx,
        fn foo(x: usize, y: usize) -> usize {
            let z = x + y;
            return z;
        }
    ).unwrap()));
}

#[test]
fn test_simplification() {
    let cx = mk_ctxt();
    let item = quote_item!(&cx,
        fn new_int_alist<B>() -> alist<isize, B> {
            fn eq_int(a: isize, b: isize) -> bool { a == b }
            return alist {eq_fn: eq_int, data: Vec::new()};
        }
    ).unwrap();
    let fnia = FakeNodeIdAssigner;
    let lcx = LoweringContext::new(&fnia, None);
    let hir_item = lower_item(&lcx, &item);
    let item_in = InlinedItemRef::Item(&hir_item);
    let item_out = simplify_ast(item_in);
    let item_exp = InlinedItem::Item(P(lower_item(&lcx, &quote_item!(&cx,
        fn new_int_alist<B>() -> alist<isize, B> {
            return alist {eq_fn: eq_int, data: Vec::new()};
        }
    ).unwrap())));
    match (item_out, item_exp) {
      (InlinedItem::Item(item_out), InlinedItem::Item(item_exp)) => {
        assert!(pprust::item_to_string(&*item_out) ==
                pprust::item_to_string(&*item_exp));
      }
      _ => panic!()
    }
}
