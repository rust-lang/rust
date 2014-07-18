// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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

use c = metadata::common;
use cstore = metadata::cstore;
use driver::session::Session;
use metadata::decoder;
use middle::def;
use e = metadata::encoder;
use middle::freevars::freevar_entry;
use middle::region;
use metadata::tydecode;
use metadata::tydecode::{DefIdSource, NominalType, TypeWithId, TypeParameter,
                         RegionParameter};
use metadata::tyencode;
use middle::subst;
use middle::subst::VecPerParamSpace;
use middle::typeck::{MethodCall, MethodCallee, MethodOrigin};
use middle::{ty, typeck};
use util::ppaux::ty_to_string;

use syntax::{ast, ast_map, ast_util, codemap, fold};
use syntax::ast_util::PostExpansionMethod;
use syntax::codemap::Span;
use syntax::fold::Folder;
use syntax::parse::token;
use syntax;

use libc;
use std::io::Seek;
use std::io::MemWriter;
use std::mem;
use std::gc::GC;

use serialize::ebml::reader;
use serialize::ebml;
use serialize;
use serialize::{Encoder, Encodable, EncoderHelpers, DecoderHelpers};
use serialize::{Decoder, Decodable};
use writer = serialize::ebml::writer;

#[cfg(test)] use syntax::parse;
#[cfg(test)] use syntax::print::pprust;
#[cfg(test)] use std::gc::Gc;

struct DecodeContext<'a> {
    cdata: &'a cstore::crate_metadata,
    tcx: &'a ty::ctxt,
}

struct ExtendedDecodeContext<'a> {
    dcx: &'a DecodeContext<'a>,
    from_id_range: ast_util::IdRange,
    to_id_range: ast_util::IdRange
}

trait tr {
    fn tr(&self, xcx: &ExtendedDecodeContext) -> Self;
}

trait tr_intern {
    fn tr_intern(&self, xcx: &ExtendedDecodeContext) -> ast::DefId;
}

pub type Encoder<'a> = writer::Encoder<'a, MemWriter>;

// ______________________________________________________________________
// Top-level methods.

pub fn encode_inlined_item(ecx: &e::EncodeContext,
                           ebml_w: &mut Encoder,
                           ii: e::InlinedItemRef) {
    let id = match ii {
        e::IIItemRef(i) => i.id,
        e::IIForeignRef(i) => i.id,
        e::IIMethodRef(_, _, m) => m.id,
    };
    debug!("> Encoding inlined item: {} ({})",
           ecx.tcx.map.path_to_string(id),
           ebml_w.writer.tell());

    let ii = simplify_ast(ii);
    let id_range = ast_util::compute_id_range_for_inlined_item(&ii);

    ebml_w.start_tag(c::tag_ast as uint);
    id_range.encode(ebml_w);
    encode_ast(ebml_w, ii);
    encode_side_tables_for_ii(ecx, ebml_w, &ii);
    ebml_w.end_tag();

    debug!("< Encoded inlined fn: {} ({})",
           ecx.tcx.map.path_to_string(id),
           ebml_w.writer.tell());
}

pub fn decode_inlined_item(cdata: &cstore::crate_metadata,
                           tcx: &ty::ctxt,
                           path: Vec<ast_map::PathElem>,
                           par_doc: ebml::Doc)
                           -> Result<ast::InlinedItem, Vec<ast_map::PathElem>> {
    let dcx = &DecodeContext {
        cdata: cdata,
        tcx: tcx,
    };
    match par_doc.opt_child(c::tag_ast) {
      None => Err(path),
      Some(ast_doc) => {
        let mut path_as_str = None;
        debug!("> Decoding inlined fn: {}::?",
        {
            // Do an Option dance to use the path after it is moved below.
            let s = ast_map::path_to_string(ast_map::Values(path.iter()));
            path_as_str = Some(s);
            path_as_str.as_ref().map(|x| x.as_slice())
        });
        let mut ast_dsr = reader::Decoder::new(ast_doc);
        let from_id_range = Decodable::decode(&mut ast_dsr).unwrap();
        let to_id_range = reserve_id_range(&dcx.tcx.sess, from_id_range);
        let xcx = &ExtendedDecodeContext {
            dcx: dcx,
            from_id_range: from_id_range,
            to_id_range: to_id_range
        };
        let raw_ii = decode_ast(ast_doc);
        let ii = renumber_and_map_ast(xcx, &dcx.tcx.map, path, raw_ii);
        let ident = match ii {
            ast::IIItem(i) => i.ident,
            ast::IIForeign(i) => i.ident,
            ast::IIMethod(_, _, m) => m.pe_ident(),
        };
        debug!("Fn named: {}", token::get_ident(ident));
        debug!("< Decoded inlined fn: {}::{}",
               path_as_str.unwrap(),
               token::get_ident(ident));
        region::resolve_inlined_item(&tcx.sess, &tcx.region_maps, &ii);
        decode_side_tables(xcx, ast_doc);
        match ii {
          ast::IIItem(i) => {
            debug!(">>> DECODED ITEM >>>\n{}\n<<< DECODED ITEM <<<",
                   syntax::print::pprust::item_to_string(&*i));
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

impl<'a> ExtendedDecodeContext<'a> {
    pub fn tr_id(&self, id: ast::NodeId) -> ast::NodeId {
        /*!
         * Translates an internal id, meaning a node id that is known
         * to refer to some part of the item currently being inlined,
         * such as a local variable or argument.  All naked node-ids
         * that appear in types have this property, since if something
         * might refer to an external item we would use a def-id to
         * allow for the possibility that the item resides in another
         * crate.
         */

        // from_id_range should be non-empty
        assert!(!self.from_id_range.empty());
        (id - self.from_id_range.min + self.to_id_range.min)
    }
    pub fn tr_def_id(&self, did: ast::DefId) -> ast::DefId {
        /*!
         * Translates an EXTERNAL def-id, converting the crate number
         * from the one used in the encoded data to the current crate
         * numbers..  By external, I mean that it be translated to a
         * reference to the item in its original crate, as opposed to
         * being translated to a reference to the inlined version of
         * the item.  This is typically, but not always, what you
         * want, because most def-ids refer to external things like
         * types or other fns that may or may not be inlined.  Note
         * that even when the inlined function is referencing itself
         * recursively, we would want `tr_def_id` for that
         * reference--- conceptually the function calls the original,
         * non-inlined version, and trans deals with linking that
         * recursive call to the inlined copy.
         *
         * However, there are a *few* cases where def-ids are used but
         * we know that the thing being referenced is in fact *internal*
         * to the item being inlined.  In those cases, you should use
         * `tr_intern_def_id()` below.
         */

        decoder::translate_def_id(self.dcx.cdata, did)
    }
    pub fn tr_intern_def_id(&self, did: ast::DefId) -> ast::DefId {
        /*!
         * Translates an INTERNAL def-id, meaning a def-id that is
         * known to refer to some part of the item currently being
         * inlined.  In that case, we want to convert the def-id to
         * refer to the current crate and to the new, inlined node-id.
         */

        assert_eq!(did.krate, ast::LOCAL_CRATE);
        ast::DefId { krate: ast::LOCAL_CRATE, node: self.tr_id(did.node) }
    }
    pub fn tr_span(&self, _span: Span) -> Span {
        codemap::DUMMY_SP // FIXME (#1972): handle span properly
    }
}

impl tr_intern for ast::DefId {
    fn tr_intern(&self, xcx: &ExtendedDecodeContext) -> ast::DefId {
        xcx.tr_intern_def_id(*self)
    }
}

impl tr for ast::DefId {
    fn tr(&self, xcx: &ExtendedDecodeContext) -> ast::DefId {
        xcx.tr_def_id(*self)
    }
}

impl tr for Option<ast::DefId> {
    fn tr(&self, xcx: &ExtendedDecodeContext) -> Option<ast::DefId> {
        self.map(|d| xcx.tr_def_id(d))
    }
}

impl tr for Span {
    fn tr(&self, xcx: &ExtendedDecodeContext) -> Span {
        xcx.tr_span(*self)
    }
}

trait def_id_encoder_helpers {
    fn emit_def_id(&mut self, did: ast::DefId);
}

impl<S:serialize::Encoder<E>, E> def_id_encoder_helpers for S {
    fn emit_def_id(&mut self, did: ast::DefId) {
        did.encode(self).ok().unwrap()
    }
}

trait def_id_decoder_helpers {
    fn read_def_id(&mut self, xcx: &ExtendedDecodeContext) -> ast::DefId;
    fn read_def_id_noxcx(&mut self,
                         cdata: &cstore::crate_metadata) -> ast::DefId;
}

impl<D:serialize::Decoder<E>, E> def_id_decoder_helpers for D {
    fn read_def_id(&mut self, xcx: &ExtendedDecodeContext) -> ast::DefId {
        let did: ast::DefId = Decodable::decode(self).ok().unwrap();
        did.tr(xcx)
    }

    fn read_def_id_noxcx(&mut self,
                         cdata: &cstore::crate_metadata) -> ast::DefId {
        let did: ast::DefId = Decodable::decode(self).ok().unwrap();
        decoder::translate_def_id(cdata, did)
    }
}

// ______________________________________________________________________
// Encoding and decoding the AST itself
//
// The hard work is done by an autogenerated module astencode_gen.  To
// regenerate astencode_gen, run src/etc/gen-astencode.  It will
// replace astencode_gen with a dummy file and regenerate its
// contents.  If you get compile errors, the dummy file
// remains---resolve the errors and then rerun astencode_gen.
// Annoying, I know, but hopefully only temporary.
//
// When decoding, we have to renumber the AST so that the node ids that
// appear within are disjoint from the node ids in our existing ASTs.
// We also have to adjust the spans: for now we just insert a dummy span,
// but eventually we should add entries to the local codemap as required.

fn encode_ast(ebml_w: &mut Encoder, item: ast::InlinedItem) {
    ebml_w.start_tag(c::tag_tree as uint);
    item.encode(ebml_w);
    ebml_w.end_tag();
}

struct NestedItemsDropper;

impl Folder for NestedItemsDropper {
    fn fold_block(&mut self, blk: ast::P<ast::Block>) -> ast::P<ast::Block> {
        let stmts_sans_items = blk.stmts.iter().filter_map(|stmt| {
            match stmt.node {
                ast::StmtExpr(_, _) | ast::StmtSemi(_, _) => Some(*stmt),
                ast::StmtDecl(decl, _) => {
                    match decl.node {
                        ast::DeclLocal(_) => Some(*stmt),
                        ast::DeclItem(_) => None,
                    }
                }
                ast::StmtMac(..) => fail!("unexpanded macro in astencode")
            }
        }).collect();
        let blk_sans_items = ast::P(ast::Block {
            view_items: Vec::new(), // I don't know if we need the view_items
                                    // here, but it doesn't break tests!
            stmts: stmts_sans_items,
            expr: blk.expr,
            id: blk.id,
            rules: blk.rules,
            span: blk.span,
        });
        fold::noop_fold_block(blk_sans_items, self)
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
fn simplify_ast(ii: e::InlinedItemRef) -> ast::InlinedItem {
    let mut fld = NestedItemsDropper;

    match ii {
        // HACK we're not dropping items.
        e::IIItemRef(i) => ast::IIItem(fold::noop_fold_item(i, &mut fld)
                                       .expect_one("expected one item")),
        e::IIMethodRef(d, p, m) => ast::IIMethod(d, p, fold::noop_fold_method(m, &mut fld)
                                                 .expect_one(
                "noop_fold_method must produce exactly one method")),
        e::IIForeignRef(i) => ast::IIForeign(fold::noop_fold_foreign_item(i, &mut fld))
    }
}

fn decode_ast(par_doc: ebml::Doc) -> ast::InlinedItem {
    let chi_doc = par_doc.get(c::tag_tree as uint);
    let mut d = reader::Decoder::new(chi_doc);
    Decodable::decode(&mut d).unwrap()
}

struct AstRenumberer<'a> {
    xcx: &'a ExtendedDecodeContext<'a>,
}

impl<'a> ast_map::FoldOps for AstRenumberer<'a> {
    fn new_id(&self, id: ast::NodeId) -> ast::NodeId {
        if id == ast::DUMMY_NODE_ID {
            // Used by ast_map to map the NodeInlinedParent.
            self.xcx.dcx.tcx.sess.next_node_id()
        } else {
            self.xcx.tr_id(id)
        }
    }
    fn new_span(&self, span: Span) -> Span {
        self.xcx.tr_span(span)
    }
}

fn renumber_and_map_ast(xcx: &ExtendedDecodeContext,
                        map: &ast_map::Map,
                        path: Vec<ast_map::PathElem> ,
                        ii: ast::InlinedItem) -> ast::InlinedItem {
    ast_map::map_decoded_item(map,
                              path.move_iter().collect(),
                              AstRenumberer { xcx: xcx },
                              |fld| {
        match ii {
            ast::IIItem(i) => {
                ast::IIItem(fld.fold_item(i).expect_one("expected one item"))
            }
            ast::IIMethod(d, is_provided, m) => {
                ast::IIMethod(xcx.tr_def_id(d), is_provided, fld.fold_method(m)
                              .expect_one("expected one method"))
            }
            ast::IIForeign(i) => ast::IIForeign(fld.fold_foreign_item(i))
        }
    })
}

// ______________________________________________________________________
// Encoding and decoding of ast::def

fn decode_def(xcx: &ExtendedDecodeContext, doc: ebml::Doc) -> def::Def {
    let mut dsr = reader::Decoder::new(doc);
    let def: def::Def = Decodable::decode(&mut dsr).unwrap();
    def.tr(xcx)
}

impl tr for def::Def {
    fn tr(&self, xcx: &ExtendedDecodeContext) -> def::Def {
        match *self {
          def::DefFn(did, p) => def::DefFn(did.tr(xcx), p),
          def::DefStaticMethod(did, wrapped_did2, p) => {
            def::DefStaticMethod(did.tr(xcx),
                                   match wrapped_did2 {
                                    def::FromTrait(did2) => {
                                        def::FromTrait(did2.tr(xcx))
                                    }
                                    def::FromImpl(did2) => {
                                        def::FromImpl(did2.tr(xcx))
                                    }
                                   },
                                   p)
          }
          def::DefMethod(did0, did1) => {
            def::DefMethod(did0.tr(xcx), did1.map(|did1| did1.tr(xcx)))
          }
          def::DefSelfTy(nid) => { def::DefSelfTy(xcx.tr_id(nid)) }
          def::DefMod(did) => { def::DefMod(did.tr(xcx)) }
          def::DefForeignMod(did) => { def::DefForeignMod(did.tr(xcx)) }
          def::DefStatic(did, m) => { def::DefStatic(did.tr(xcx), m) }
          def::DefArg(nid, b) => { def::DefArg(xcx.tr_id(nid), b) }
          def::DefLocal(nid, b) => { def::DefLocal(xcx.tr_id(nid), b) }
          def::DefVariant(e_did, v_did, is_s) => {
            def::DefVariant(e_did.tr(xcx), v_did.tr(xcx), is_s)
          },
          def::DefTrait(did) => def::DefTrait(did.tr(xcx)),
          def::DefTy(did) => def::DefTy(did.tr(xcx)),
          def::DefPrimTy(p) => def::DefPrimTy(p),
          def::DefTyParam(s, did, v) => def::DefTyParam(s, did.tr(xcx), v),
          def::DefBinding(nid, bm) => def::DefBinding(xcx.tr_id(nid), bm),
          def::DefUse(did) => def::DefUse(did.tr(xcx)),
          def::DefUpvar(nid1, def, nid2, nid3) => {
            def::DefUpvar(xcx.tr_id(nid1),
                           box(GC) (*def).tr(xcx),
                           xcx.tr_id(nid2),
                           xcx.tr_id(nid3))
          }
          def::DefStruct(did) => def::DefStruct(did.tr(xcx)),
          def::DefRegion(nid) => def::DefRegion(xcx.tr_id(nid)),
          def::DefTyParamBinder(nid) => {
            def::DefTyParamBinder(xcx.tr_id(nid))
          }
          def::DefLabel(nid) => def::DefLabel(xcx.tr_id(nid))
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of adjustment information

impl tr for ty::AutoDerefRef {
    fn tr(&self, xcx: &ExtendedDecodeContext) -> ty::AutoDerefRef {
        ty::AutoDerefRef {
            autoderefs: self.autoderefs,
            autoref: match self.autoref {
                Some(ref autoref) => Some(autoref.tr(xcx)),
                None => None
            }
        }
    }
}

impl tr for ty::AutoRef {
    fn tr(&self, xcx: &ExtendedDecodeContext) -> ty::AutoRef {
        self.map_region(|r| r.tr(xcx))
    }
}

impl tr for ty::Region {
    fn tr(&self, xcx: &ExtendedDecodeContext) -> ty::Region {
        match *self {
            ty::ReLateBound(id, br) => {
                ty::ReLateBound(xcx.tr_id(id), br.tr(xcx))
            }
            ty::ReEarlyBound(id, space, index, ident) => {
                ty::ReEarlyBound(xcx.tr_id(id), space, index, ident)
            }
            ty::ReScope(id) => {
                ty::ReScope(xcx.tr_id(id))
            }
            ty::ReEmpty | ty::ReStatic | ty::ReInfer(..) => {
                *self
            }
            ty::ReFree(ref fr) => {
                ty::ReFree(ty::FreeRegion {scope_id: xcx.tr_id(fr.scope_id),
                                            bound_region: fr.bound_region.tr(xcx)})
            }
        }
    }
}

impl tr for ty::BoundRegion {
    fn tr(&self, xcx: &ExtendedDecodeContext) -> ty::BoundRegion {
        match *self {
            ty::BrAnon(_) |
            ty::BrFresh(_) => *self,
            ty::BrNamed(id, ident) => ty::BrNamed(xcx.tr_def_id(id),
                                                    ident),
        }
    }
}

impl tr for ty::TraitStore {
    fn tr(&self, xcx: &ExtendedDecodeContext) -> ty::TraitStore {
        match *self {
            ty::RegionTraitStore(r, m) => {
                ty::RegionTraitStore(r.tr(xcx), m)
            }
            ty::UniqTraitStore => ty::UniqTraitStore
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of freevar information

fn encode_freevar_entry(ebml_w: &mut Encoder, fv: &freevar_entry) {
    (*fv).encode(ebml_w).unwrap();
}

trait ebml_decoder_helper {
    fn read_freevar_entry(&mut self, xcx: &ExtendedDecodeContext)
                          -> freevar_entry;
}

impl<'a> ebml_decoder_helper for reader::Decoder<'a> {
    fn read_freevar_entry(&mut self, xcx: &ExtendedDecodeContext)
                          -> freevar_entry {
        let fv: freevar_entry = Decodable::decode(self).unwrap();
        fv.tr(xcx)
    }
}

impl tr for freevar_entry {
    fn tr(&self, xcx: &ExtendedDecodeContext) -> freevar_entry {
        freevar_entry {
            def: self.def.tr(xcx),
            span: self.span.tr(xcx),
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of MethodCallee

trait read_method_callee_helper {
    fn read_method_callee(&mut self, xcx: &ExtendedDecodeContext)
        -> (typeck::ExprAdjustment, MethodCallee);
}

fn encode_method_callee(ecx: &e::EncodeContext,
                        ebml_w: &mut Encoder,
                        adjustment: typeck::ExprAdjustment,
                        method: &MethodCallee) {
    ebml_w.emit_struct("MethodCallee", 4, |ebml_w| {
        ebml_w.emit_struct_field("adjustment", 0u, |ebml_w| {
            adjustment.encode(ebml_w)
        });
        ebml_w.emit_struct_field("origin", 1u, |ebml_w| {
            method.origin.encode(ebml_w)
        });
        ebml_w.emit_struct_field("ty", 2u, |ebml_w| {
            Ok(ebml_w.emit_ty(ecx, method.ty))
        });
        ebml_w.emit_struct_field("substs", 3u, |ebml_w| {
            Ok(ebml_w.emit_substs(ecx, &method.substs))
        })
    }).unwrap();
}

impl<'a> read_method_callee_helper for reader::Decoder<'a> {
    fn read_method_callee(&mut self, xcx: &ExtendedDecodeContext)
        -> (typeck::ExprAdjustment, MethodCallee) {

        self.read_struct("MethodCallee", 4, |this| {
            let adjustment = this.read_struct_field("adjustment", 0, |this| {
                Decodable::decode(this)
            }).unwrap();
            Ok((adjustment, MethodCallee {
                origin: this.read_struct_field("origin", 1, |this| {
                    let method_origin: MethodOrigin =
                        Decodable::decode(this).unwrap();
                    Ok(method_origin.tr(xcx))
                }).unwrap(),
                ty: this.read_struct_field("ty", 2, |this| {
                    Ok(this.read_ty(xcx))
                }).unwrap(),
                substs: this.read_struct_field("substs", 3, |this| {
                    Ok(this.read_substs(xcx))
                }).unwrap()
            }))
        }).unwrap()
    }
}

impl tr for MethodOrigin {
    fn tr(&self, xcx: &ExtendedDecodeContext) -> MethodOrigin {
        match *self {
            typeck::MethodStatic(did) => typeck::MethodStatic(did.tr(xcx)),
            typeck::MethodStaticUnboxedClosure(did) => {
                typeck::MethodStaticUnboxedClosure(did.tr(xcx))
            }
            typeck::MethodParam(ref mp) => {
                typeck::MethodParam(
                    typeck::MethodParam {
                        trait_id: mp.trait_id.tr(xcx),
                        .. *mp
                    }
                )
            }
            typeck::MethodObject(ref mo) => {
                typeck::MethodObject(
                    typeck::MethodObject {
                        trait_id: mo.trait_id.tr(xcx),
                        .. *mo
                    }
                )
            }
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding vtable_res

fn encode_vtable_res_with_key(ecx: &e::EncodeContext,
                              ebml_w: &mut Encoder,
                              adjustment: typeck::ExprAdjustment,
                              dr: &typeck::vtable_res) {
    ebml_w.emit_struct("VtableWithKey", 2, |ebml_w| {
        ebml_w.emit_struct_field("adjustment", 0u, |ebml_w| {
            adjustment.encode(ebml_w)
        });
        ebml_w.emit_struct_field("vtable_res", 1u, |ebml_w| {
            Ok(encode_vtable_res(ecx, ebml_w, dr))
        })
    }).unwrap()
}

pub fn encode_vtable_res(ecx: &e::EncodeContext,
                         ebml_w: &mut Encoder,
                         dr: &typeck::vtable_res) {
    // can't autogenerate this code because automatic code of
    // ty::t doesn't work, and there is no way (atm) to have
    // hand-written encoding routines combine with auto-generated
    // ones. perhaps we should fix this.
    encode_vec_per_param_space(
        ebml_w, dr,
        |ebml_w, param_tables| encode_vtable_param_res(ecx, ebml_w,
                                                       param_tables))
}

pub fn encode_vtable_param_res(ecx: &e::EncodeContext,
                     ebml_w: &mut Encoder,
                     param_tables: &typeck::vtable_param_res) {
    ebml_w.emit_from_vec(param_tables.as_slice(), |ebml_w, vtable_origin| {
        Ok(encode_vtable_origin(ecx, ebml_w, vtable_origin))
    }).unwrap()
}


pub fn encode_vtable_origin(ecx: &e::EncodeContext,
                        ebml_w: &mut Encoder,
                        vtable_origin: &typeck::vtable_origin) {
    ebml_w.emit_enum("vtable_origin", |ebml_w| {
        match *vtable_origin {
          typeck::vtable_static(def_id, ref substs, ref vtable_res) => {
            ebml_w.emit_enum_variant("vtable_static", 0u, 3u, |ebml_w| {
                ebml_w.emit_enum_variant_arg(0u, |ebml_w| {
                    Ok(ebml_w.emit_def_id(def_id))
                });
                ebml_w.emit_enum_variant_arg(1u, |ebml_w| {
                    Ok(ebml_w.emit_substs(ecx, substs))
                });
                ebml_w.emit_enum_variant_arg(2u, |ebml_w| {
                    Ok(encode_vtable_res(ecx, ebml_w, vtable_res))
                })
            })
          }
          typeck::vtable_param(pn, bn) => {
            ebml_w.emit_enum_variant("vtable_param", 1u, 3u, |ebml_w| {
                ebml_w.emit_enum_variant_arg(0u, |ebml_w| {
                    pn.encode(ebml_w)
                });
                ebml_w.emit_enum_variant_arg(1u, |ebml_w| {
                    ebml_w.emit_uint(bn)
                })
            })
          }
          typeck::vtable_unboxed_closure(def_id) => {
              ebml_w.emit_enum_variant("vtable_unboxed_closure",
                                       2u,
                                       1u,
                                       |ebml_w| {
                ebml_w.emit_enum_variant_arg(0u, |ebml_w| {
                    Ok(ebml_w.emit_def_id(def_id))
                })
              })
          }
          typeck::vtable_error => {
            ebml_w.emit_enum_variant("vtable_error", 3u, 3u, |_ebml_w| {
                Ok(())
            })
          }
        }
    }).unwrap()
}

pub trait vtable_decoder_helpers {
    fn read_vec_per_param_space<T>(&mut self,
                                   f: |&mut Self| -> T)
                                   -> VecPerParamSpace<T>;
    fn read_vtable_res_with_key(&mut self,
                                tcx: &ty::ctxt,
                                cdata: &cstore::crate_metadata)
                                -> (typeck::ExprAdjustment, typeck::vtable_res);
    fn read_vtable_res(&mut self,
                       tcx: &ty::ctxt, cdata: &cstore::crate_metadata)
                      -> typeck::vtable_res;
    fn read_vtable_param_res(&mut self,
                       tcx: &ty::ctxt, cdata: &cstore::crate_metadata)
                      -> typeck::vtable_param_res;
    fn read_vtable_origin(&mut self,
                          tcx: &ty::ctxt, cdata: &cstore::crate_metadata)
                          -> typeck::vtable_origin;
}

impl<'a> vtable_decoder_helpers for reader::Decoder<'a> {
    fn read_vec_per_param_space<T>(&mut self,
                                   f: |&mut reader::Decoder<'a>| -> T)
                                   -> VecPerParamSpace<T>
    {
        let types = self.read_to_vec(|this| Ok(f(this))).unwrap();
        let selfs = self.read_to_vec(|this| Ok(f(this))).unwrap();
        let fns = self.read_to_vec(|this| Ok(f(this))).unwrap();
        VecPerParamSpace::new(types, selfs, fns)
    }

    fn read_vtable_res_with_key(&mut self,
                                tcx: &ty::ctxt,
                                cdata: &cstore::crate_metadata)
                                -> (typeck::ExprAdjustment, typeck::vtable_res) {
        self.read_struct("VtableWithKey", 2, |this| {
            let adjustment = this.read_struct_field("adjustment", 0, |this| {
                Decodable::decode(this)
            }).unwrap();
            Ok((adjustment, this.read_struct_field("vtable_res", 1, |this| {
                Ok(this.read_vtable_res(tcx, cdata))
            }).unwrap()))
        }).unwrap()
    }

    fn read_vtable_res(&mut self,
                       tcx: &ty::ctxt,
                       cdata: &cstore::crate_metadata)
                       -> typeck::vtable_res
    {
        self.read_vec_per_param_space(
            |this| this.read_vtable_param_res(tcx, cdata))
    }

    fn read_vtable_param_res(&mut self,
                             tcx: &ty::ctxt, cdata: &cstore::crate_metadata)
                      -> typeck::vtable_param_res {
        self.read_to_vec(|this| Ok(this.read_vtable_origin(tcx, cdata)))
             .unwrap().move_iter().collect()
    }

    fn read_vtable_origin(&mut self,
                          tcx: &ty::ctxt, cdata: &cstore::crate_metadata)
        -> typeck::vtable_origin {
        self.read_enum("vtable_origin", |this| {
            this.read_enum_variant(["vtable_static",
                                    "vtable_param",
                                    "vtable_error",
                                    "vtable_unboxed_closure"],
                                   |this, i| {
                Ok(match i {
                  0 => {
                    typeck::vtable_static(
                        this.read_enum_variant_arg(0u, |this| {
                            Ok(this.read_def_id_noxcx(cdata))
                        }).unwrap(),
                        this.read_enum_variant_arg(1u, |this| {
                            Ok(this.read_substs_noxcx(tcx, cdata))
                        }).unwrap(),
                        this.read_enum_variant_arg(2u, |this| {
                            Ok(this.read_vtable_res(tcx, cdata))
                        }).unwrap()
                    )
                  }
                  1 => {
                    typeck::vtable_param(
                        this.read_enum_variant_arg(0u, |this| {
                            Decodable::decode(this)
                        }).unwrap(),
                        this.read_enum_variant_arg(1u, |this| {
                            this.read_uint()
                        }).unwrap()
                    )
                  }
                  2 => {
                    typeck::vtable_unboxed_closure(
                        this.read_enum_variant_arg(0u, |this| {
                            Ok(this.read_def_id_noxcx(cdata))
                        }).unwrap()
                    )
                  }
                  3 => {
                    typeck::vtable_error
                  }
                  _ => fail!("bad enum variant")
                })
            })
        }).unwrap()
    }
}

// ___________________________________________________________________________
//

fn encode_vec_per_param_space<T>(ebml_w: &mut Encoder,
                                 v: &subst::VecPerParamSpace<T>,
                                 f: |&mut Encoder, &T|) {
    for &space in subst::ParamSpace::all().iter() {
        ebml_w.emit_from_vec(v.get_slice(space),
                             |ebml_w, n| Ok(f(ebml_w, n))).unwrap();
    }
}

// ______________________________________________________________________
// Encoding and decoding the side tables

trait get_ty_str_ctxt {
    fn ty_str_ctxt<'a>(&'a self) -> tyencode::ctxt<'a>;
}

impl<'a> get_ty_str_ctxt for e::EncodeContext<'a> {
    fn ty_str_ctxt<'a>(&'a self) -> tyencode::ctxt<'a> {
        tyencode::ctxt {
            diag: self.tcx.sess.diagnostic(),
            ds: e::def_to_string,
            tcx: self.tcx,
            abbrevs: &self.type_abbrevs
        }
    }
}

trait ebml_writer_helpers {
    fn emit_closure_type(&mut self,
                         ecx: &e::EncodeContext,
                         closure_type: &ty::ClosureTy);
    fn emit_ty(&mut self, ecx: &e::EncodeContext, ty: ty::t);
    fn emit_tys(&mut self, ecx: &e::EncodeContext, tys: &[ty::t]);
    fn emit_type_param_def(&mut self,
                           ecx: &e::EncodeContext,
                           type_param_def: &ty::TypeParameterDef);
    fn emit_polytype(&mut self,
                     ecx: &e::EncodeContext,
                     pty: ty::Polytype);
    fn emit_substs(&mut self, ecx: &e::EncodeContext, substs: &subst::Substs);
    fn emit_auto_adjustment(&mut self, ecx: &e::EncodeContext, adj: &ty::AutoAdjustment);
}

impl<'a> ebml_writer_helpers for Encoder<'a> {
    fn emit_closure_type(&mut self,
                         ecx: &e::EncodeContext,
                         closure_type: &ty::ClosureTy) {
        self.emit_opaque(|this| {
            Ok(e::write_closure_type(ecx, this, closure_type))
        });
    }

    fn emit_ty(&mut self, ecx: &e::EncodeContext, ty: ty::t) {
        self.emit_opaque(|this| Ok(e::write_type(ecx, this, ty)));
    }

    fn emit_tys(&mut self, ecx: &e::EncodeContext, tys: &[ty::t]) {
        self.emit_from_vec(tys, |this, ty| Ok(this.emit_ty(ecx, *ty)));
    }

    fn emit_type_param_def(&mut self,
                           ecx: &e::EncodeContext,
                           type_param_def: &ty::TypeParameterDef) {
        self.emit_opaque(|this| {
            Ok(tyencode::enc_type_param_def(this.writer,
                                         &ecx.ty_str_ctxt(),
                                         type_param_def))
        });
    }

    fn emit_polytype(&mut self,
                 ecx: &e::EncodeContext,
                 pty: ty::Polytype) {
        self.emit_struct("Polytype", 2, |this| {
            this.emit_struct_field("generics", 0, |this| {
                this.emit_struct("Generics", 2, |this| {
                    this.emit_struct_field("types", 0, |this| {
                        Ok(encode_vec_per_param_space(
                            this, &pty.generics.types,
                            |this, def| this.emit_type_param_def(ecx, def)))
                    });
                    this.emit_struct_field("regions", 1, |this| {
                        Ok(encode_vec_per_param_space(
                            this, &pty.generics.regions,
                            |this, def| def.encode(this).unwrap()))
                    })
                })
            });
            this.emit_struct_field("ty", 1, |this| {
                Ok(this.emit_ty(ecx, pty.ty))
            })
        });
    }

    fn emit_substs(&mut self, ecx: &e::EncodeContext, substs: &subst::Substs) {
        self.emit_opaque(|this| Ok(tyencode::enc_substs(this.writer,
                                                           &ecx.ty_str_ctxt(),
                                                           substs)));
    }

    fn emit_auto_adjustment(&mut self, ecx: &e::EncodeContext, adj: &ty::AutoAdjustment) {
        self.emit_enum("AutoAdjustment", |this| {
            match *adj {
                ty::AutoAddEnv(store) => {
                    this.emit_enum_variant("AutoAddEnv", 0, 1, |this| {
                        this.emit_enum_variant_arg(0, |this| store.encode(this))
                    })
                }

                ty::AutoDerefRef(ref auto_deref_ref) => {
                    this.emit_enum_variant("AutoDerefRef", 1, 1, |this| {
                        this.emit_enum_variant_arg(0, |this| auto_deref_ref.encode(this))
                    })
                }

                ty::AutoObject(store, b, def_id, ref substs) => {
                    this.emit_enum_variant("AutoObject", 2, 4, |this| {
                        this.emit_enum_variant_arg(0, |this| store.encode(this));
                        this.emit_enum_variant_arg(1, |this| b.encode(this));
                        this.emit_enum_variant_arg(2, |this| def_id.encode(this));
                        this.emit_enum_variant_arg(3, |this| Ok(this.emit_substs(ecx, substs)))
                    })
                }
            }
        });
    }
}

trait write_tag_and_id {
    fn tag(&mut self, tag_id: c::astencode_tag, f: |&mut Self|);
    fn id(&mut self, id: ast::NodeId);
}

impl<'a> write_tag_and_id for Encoder<'a> {
    fn tag(&mut self,
           tag_id: c::astencode_tag,
           f: |&mut Encoder<'a>|) {
        self.start_tag(tag_id as uint);
        f(self);
        self.end_tag();
    }

    fn id(&mut self, id: ast::NodeId) {
        self.wr_tagged_u64(c::tag_table_id as uint, id as u64);
    }
}

struct SideTableEncodingIdVisitor<'a,'b> {
    ecx_ptr: *const libc::c_void,
    new_ebml_w: &'a mut Encoder<'b>,
}

impl<'a,'b> ast_util::IdVisitingOperation for
        SideTableEncodingIdVisitor<'a,'b> {
    fn visit_id(&self, id: ast::NodeId) {
        // Note: this will cause a copy of ebml_w, which is bad as
        // it is mutable. But I believe it's harmless since we generate
        // balanced EBML.
        //
        // FIXME(pcwalton): Don't copy this way.
        let mut new_ebml_w = unsafe {
            self.new_ebml_w.unsafe_clone()
        };
        // See above
        let ecx: &e::EncodeContext = unsafe {
            mem::transmute(self.ecx_ptr)
        };
        encode_side_tables_for_id(ecx, &mut new_ebml_w, id)
    }
}

fn encode_side_tables_for_ii(ecx: &e::EncodeContext,
                             ebml_w: &mut Encoder,
                             ii: &ast::InlinedItem) {
    ebml_w.start_tag(c::tag_table as uint);
    let mut new_ebml_w = unsafe {
        ebml_w.unsafe_clone()
    };

    // Because the ast visitor uses @IdVisitingOperation, I can't pass in
    // ecx directly, but /I/ know that it'll be fine since the lifetime is
    // tied to the CrateContext that lives throughout this entire section.
    ast_util::visit_ids_for_inlined_item(ii, &SideTableEncodingIdVisitor {
        ecx_ptr: unsafe {
            mem::transmute(ecx)
        },
        new_ebml_w: &mut new_ebml_w,
    });
    ebml_w.end_tag();
}

fn encode_side_tables_for_id(ecx: &e::EncodeContext,
                             ebml_w: &mut Encoder,
                             id: ast::NodeId) {
    let tcx = ecx.tcx;

    debug!("Encoding side tables for id {}", id);

    for def in tcx.def_map.borrow().find(&id).iter() {
        ebml_w.tag(c::tag_table_def, |ebml_w| {
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val, |ebml_w| (*def).encode(ebml_w).unwrap());
        })
    }

    for &ty in tcx.node_types.borrow().find(&(id as uint)).iter() {
        ebml_w.tag(c::tag_table_node_type, |ebml_w| {
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val, |ebml_w| {
                ebml_w.emit_ty(ecx, *ty);
            })
        })
    }

    for &item_substs in tcx.item_substs.borrow().find(&id).iter() {
        ebml_w.tag(c::tag_table_item_subst, |ebml_w| {
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val, |ebml_w| {
                ebml_w.emit_substs(ecx, &item_substs.substs);
            })
        })
    }

    for &fv in tcx.freevars.borrow().find(&id).iter() {
        ebml_w.tag(c::tag_table_freevars, |ebml_w| {
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val, |ebml_w| {
                ebml_w.emit_from_vec(fv.as_slice(), |ebml_w, fv_entry| {
                    Ok(encode_freevar_entry(ebml_w, fv_entry))
                });
            })
        })
    }

    let lid = ast::DefId { krate: ast::LOCAL_CRATE, node: id };
    for &pty in tcx.tcache.borrow().find(&lid).iter() {
        ebml_w.tag(c::tag_table_tcache, |ebml_w| {
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val, |ebml_w| {
                ebml_w.emit_polytype(ecx, pty.clone());
            })
        })
    }

    for &type_param_def in tcx.ty_param_defs.borrow().find(&id).iter() {
        ebml_w.tag(c::tag_table_param_defs, |ebml_w| {
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val, |ebml_w| {
                ebml_w.emit_type_param_def(ecx, type_param_def)
            })
        })
    }

    let method_call = MethodCall::expr(id);
    for &method in tcx.method_map.borrow().find(&method_call).iter() {
        ebml_w.tag(c::tag_table_method_map, |ebml_w| {
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val, |ebml_w| {
                encode_method_callee(ecx, ebml_w, method_call.adjustment, method)
            })
        })
    }

    for &dr in tcx.vtable_map.borrow().find(&method_call).iter() {
        ebml_w.tag(c::tag_table_vtable_map, |ebml_w| {
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val, |ebml_w| {
                encode_vtable_res_with_key(ecx, ebml_w, method_call.adjustment, dr);
            })
        })
    }

    for &adj in tcx.adjustments.borrow().find(&id).iter() {
        match *adj {
            ty::AutoDerefRef(adj) => {
                for autoderef in range(0, adj.autoderefs) {
                    let method_call = MethodCall::autoderef(id, autoderef);
                    for &method in tcx.method_map.borrow().find(&method_call).iter() {
                        ebml_w.tag(c::tag_table_method_map, |ebml_w| {
                            ebml_w.id(id);
                            ebml_w.tag(c::tag_table_val, |ebml_w| {
                                encode_method_callee(ecx, ebml_w,
                                                     method_call.adjustment, method)
                            })
                        })
                    }

                    for &dr in tcx.vtable_map.borrow().find(&method_call).iter() {
                        ebml_w.tag(c::tag_table_vtable_map, |ebml_w| {
                            ebml_w.id(id);
                            ebml_w.tag(c::tag_table_val, |ebml_w| {
                                encode_vtable_res_with_key(ecx, ebml_w,
                                                           method_call.adjustment, dr);
                            })
                        })
                    }
                }
            }
            ty::AutoObject(..) => {
                let method_call = MethodCall::autoobject(id);
                for &method in tcx.method_map.borrow().find(&method_call).iter() {
                    ebml_w.tag(c::tag_table_method_map, |ebml_w| {
                        ebml_w.id(id);
                        ebml_w.tag(c::tag_table_val, |ebml_w| {
                            encode_method_callee(ecx, ebml_w, method_call.adjustment, method)
                        })
                    })
                }

                for &dr in tcx.vtable_map.borrow().find(&method_call).iter() {
                    ebml_w.tag(c::tag_table_vtable_map, |ebml_w| {
                        ebml_w.id(id);
                        ebml_w.tag(c::tag_table_val, |ebml_w| {
                            encode_vtable_res_with_key(ecx, ebml_w, method_call.adjustment, dr);
                        })
                    })
                }
            }
            _ => {}
        }

        ebml_w.tag(c::tag_table_adjustments, |ebml_w| {
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val, |ebml_w| {
                ebml_w.emit_auto_adjustment(ecx, adj);
            })
        })
    }

    for unboxed_closure_type in tcx.unboxed_closure_types
                                   .borrow()
                                   .find(&ast_util::local_def(id))
                                   .iter() {
        ebml_w.tag(c::tag_table_unboxed_closure_type, |ebml_w| {
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val, |ebml_w| {
                ebml_w.emit_closure_type(ecx, *unboxed_closure_type)
            })
        })
    }
}

trait doc_decoder_helpers {
    fn as_int(&self) -> int;
    fn opt_child(&self, tag: c::astencode_tag) -> Option<Self>;
}

impl<'a> doc_decoder_helpers for ebml::Doc<'a> {
    fn as_int(&self) -> int { reader::doc_as_u64(*self) as int }
    fn opt_child(&self, tag: c::astencode_tag) -> Option<ebml::Doc<'a>> {
        reader::maybe_get_doc(*self, tag as uint)
    }
}

trait ebml_decoder_decoder_helpers {
    fn read_ty(&mut self, xcx: &ExtendedDecodeContext) -> ty::t;
    fn read_tys(&mut self, xcx: &ExtendedDecodeContext) -> Vec<ty::t>;
    fn read_type_param_def(&mut self, xcx: &ExtendedDecodeContext)
                           -> ty::TypeParameterDef;
    fn read_polytype(&mut self, xcx: &ExtendedDecodeContext)
                     -> ty::Polytype;
    fn read_substs(&mut self, xcx: &ExtendedDecodeContext) -> subst::Substs;
    fn read_auto_adjustment(&mut self, xcx: &ExtendedDecodeContext) -> ty::AutoAdjustment;
    fn read_unboxed_closure_type(&mut self, xcx: &ExtendedDecodeContext)
                                 -> ty::ClosureTy;
    fn convert_def_id(&mut self,
                      xcx: &ExtendedDecodeContext,
                      source: DefIdSource,
                      did: ast::DefId)
                      -> ast::DefId;

    // Versions of the type reading functions that don't need the full
    // ExtendedDecodeContext.
    fn read_ty_noxcx(&mut self,
                     tcx: &ty::ctxt, cdata: &cstore::crate_metadata) -> ty::t;
    fn read_tys_noxcx(&mut self,
                      tcx: &ty::ctxt,
                      cdata: &cstore::crate_metadata) -> Vec<ty::t>;
    fn read_substs_noxcx(&mut self, tcx: &ty::ctxt,
                         cdata: &cstore::crate_metadata)
                         -> subst::Substs;
}

impl<'a> ebml_decoder_decoder_helpers for reader::Decoder<'a> {
    fn read_ty_noxcx(&mut self,
                     tcx: &ty::ctxt, cdata: &cstore::crate_metadata) -> ty::t {
        self.read_opaque(|_, doc| {
            Ok(tydecode::parse_ty_data(
                doc.data,
                cdata.cnum,
                doc.start,
                tcx,
                |_, id| decoder::translate_def_id(cdata, id)))
        }).unwrap()
    }

    fn read_tys_noxcx(&mut self,
                      tcx: &ty::ctxt,
                      cdata: &cstore::crate_metadata) -> Vec<ty::t> {
        self.read_to_vec(|this| Ok(this.read_ty_noxcx(tcx, cdata)) )
            .unwrap()
            .move_iter()
            .collect()
    }

    fn read_substs_noxcx(&mut self,
                         tcx: &ty::ctxt,
                         cdata: &cstore::crate_metadata)
                         -> subst::Substs
    {
        self.read_opaque(|_, doc| {
            Ok(tydecode::parse_substs_data(
                doc.data,
                cdata.cnum,
                doc.start,
                tcx,
                |_, id| decoder::translate_def_id(cdata, id)))
        }).unwrap()
    }

    fn read_ty(&mut self, xcx: &ExtendedDecodeContext) -> ty::t {
        // Note: regions types embed local node ids.  In principle, we
        // should translate these node ids into the new decode
        // context.  However, we do not bother, because region types
        // are not used during trans.

        return self.read_opaque(|this, doc| {
            debug!("read_ty({})", type_string(doc));

            let ty = tydecode::parse_ty_data(
                doc.data,
                xcx.dcx.cdata.cnum,
                doc.start,
                xcx.dcx.tcx,
                |s, a| this.convert_def_id(xcx, s, a));

            Ok(ty)
        }).unwrap();

        fn type_string(doc: ebml::Doc) -> String {
            let mut str = String::new();
            for i in range(doc.start, doc.end) {
                str.push_char(doc.data[i] as char);
            }
            str
        }
    }

    fn read_tys(&mut self, xcx: &ExtendedDecodeContext) -> Vec<ty::t> {
        self.read_to_vec(|this| Ok(this.read_ty(xcx))).unwrap().move_iter().collect()
    }

    fn read_type_param_def(&mut self, xcx: &ExtendedDecodeContext)
                           -> ty::TypeParameterDef {
        self.read_opaque(|this, doc| {
            Ok(tydecode::parse_type_param_def_data(
                doc.data,
                doc.start,
                xcx.dcx.cdata.cnum,
                xcx.dcx.tcx,
                |s, a| this.convert_def_id(xcx, s, a)))
        }).unwrap()
    }

    fn read_polytype(&mut self, xcx: &ExtendedDecodeContext)
                                   -> ty::Polytype {
        self.read_struct("Polytype", 2, |this| {
            Ok(ty::Polytype {
                generics: this.read_struct_field("generics", 0, |this| {
                    this.read_struct("Generics", 2, |this| {
                        Ok(ty::Generics {
                            types:
                            this.read_struct_field("types", 0, |this| {
                                Ok(this.read_vec_per_param_space(
                                    |this| this.read_type_param_def(xcx)))
                            }).unwrap(),

                            regions:
                            this.read_struct_field("regions", 1, |this| {
                                Ok(this.read_vec_per_param_space(
                                    |this| Decodable::decode(this).unwrap()))
                            }).unwrap()
                        })
                    })
                }).unwrap(),
                ty: this.read_struct_field("ty", 1, |this| {
                    Ok(this.read_ty(xcx))
                }).unwrap()
            })
        }).unwrap()
    }

    fn read_substs(&mut self, xcx: &ExtendedDecodeContext) -> subst::Substs {
        self.read_opaque(|this, doc| {
            Ok(tydecode::parse_substs_data(doc.data,
                                        xcx.dcx.cdata.cnum,
                                        doc.start,
                                        xcx.dcx.tcx,
                                        |s, a| this.convert_def_id(xcx, s, a)))
        }).unwrap()
    }

    fn read_auto_adjustment(&mut self, xcx: &ExtendedDecodeContext) -> ty::AutoAdjustment {
        self.read_enum("AutoAdjustment", |this| {
            let variants = ["AutoAddEnv", "AutoDerefRef", "AutoObject"];
            this.read_enum_variant(variants, |this, i| {
                Ok(match i {
                    0 => {
                        let store: ty::TraitStore =
                            this.read_enum_variant_arg(0, |this| Decodable::decode(this)).unwrap();

                        ty:: AutoAddEnv(store.tr(xcx))
                    }
                    1 => {
                        let auto_deref_ref: ty::AutoDerefRef =
                            this.read_enum_variant_arg(0, |this| Decodable::decode(this)).unwrap();

                        ty::AutoDerefRef(auto_deref_ref.tr(xcx))
                    }
                    2 => {
                        let store: ty::TraitStore =
                            this.read_enum_variant_arg(0, |this| Decodable::decode(this)).unwrap();
                        let b: ty::BuiltinBounds =
                            this.read_enum_variant_arg(1, |this| Decodable::decode(this)).unwrap();
                        let def_id: ast::DefId =
                            this.read_enum_variant_arg(2, |this| Decodable::decode(this)).unwrap();
                        let substs = this.read_enum_variant_arg(3, |this| Ok(this.read_substs(xcx)))
                                    .unwrap();

                        ty::AutoObject(store.tr(xcx), b, def_id.tr(xcx), substs)
                    }
                    _ => fail!("bad enum variant for ty::AutoAdjustment")
                })
            })
        }).unwrap()
    }

    fn read_unboxed_closure_type(&mut self, xcx: &ExtendedDecodeContext)
                                 -> ty::ClosureTy {
        self.read_opaque(|this, doc| {
            Ok(tydecode::parse_ty_closure_data(
                doc.data,
                xcx.dcx.cdata.cnum,
                doc.start,
                xcx.dcx.tcx,
                |s, a| this.convert_def_id(xcx, s, a)))
        }).unwrap()
    }

    fn convert_def_id(&mut self,
                      xcx: &ExtendedDecodeContext,
                      source: tydecode::DefIdSource,
                      did: ast::DefId)
                      -> ast::DefId {
        /*!
         * Converts a def-id that appears in a type.  The correct
         * translation will depend on what kind of def-id this is.
         * This is a subtle point: type definitions are not
         * inlined into the current crate, so if the def-id names
         * a nominal type or type alias, then it should be
         * translated to refer to the source crate.
         *
         * However, *type parameters* are cloned along with the function
         * they are attached to.  So we should translate those def-ids
         * to refer to the new, cloned copy of the type parameter.
         * We only see references to free type parameters in the body of
         * an inlined function. In such cases, we need the def-id to
         * be a local id so that the TypeContents code is able to lookup
         * the relevant info in the ty_param_defs table.
         *
         * *Region parameters*, unfortunately, are another kettle of fish.
         * In such cases, def_id's can appear in types to distinguish
         * shadowed bound regions and so forth. It doesn't actually
         * matter so much what we do to these, since regions are erased
         * at trans time, but it's good to keep them consistent just in
         * case. We translate them with `tr_def_id()` which will map
         * the crate numbers back to the original source crate.
         *
         * It'd be really nice to refactor the type repr to not include
         * def-ids so that all these distinctions were unnecessary.
         */

        let r = match source {
            NominalType | TypeWithId | RegionParameter => xcx.tr_def_id(did),
            TypeParameter => xcx.tr_intern_def_id(did)
        };
        debug!("convert_def_id(source={:?}, did={:?})={:?}", source, did, r);
        return r;
    }
}

fn decode_side_tables(xcx: &ExtendedDecodeContext,
                      ast_doc: ebml::Doc) {
    let dcx = xcx.dcx;
    let tbl_doc = ast_doc.get(c::tag_table as uint);
    reader::docs(tbl_doc, |tag, entry_doc| {
        let id0 = entry_doc.get(c::tag_table_id as uint).as_int();
        let id = xcx.tr_id(id0 as ast::NodeId);

        debug!(">> Side table document with tag 0x{:x} \
                found for id {} (orig {})",
               tag, id, id0);

        match c::astencode_tag::from_uint(tag) {
            None => {
                xcx.dcx.tcx.sess.bug(
                    format!("unknown tag found in side tables: {:x}",
                            tag).as_slice());
            }
            Some(value) => {
                let val_doc = entry_doc.get(c::tag_table_val as uint);
                let mut val_dsr = reader::Decoder::new(val_doc);
                let val_dsr = &mut val_dsr;

                match value {
                    c::tag_table_def => {
                        let def = decode_def(xcx, val_doc);
                        dcx.tcx.def_map.borrow_mut().insert(id, def);
                    }
                    c::tag_table_node_type => {
                        let ty = val_dsr.read_ty(xcx);
                        debug!("inserting ty for node {:?}: {}",
                               id, ty_to_string(dcx.tcx, ty));
                        dcx.tcx.node_types.borrow_mut().insert(id as uint, ty);
                    }
                    c::tag_table_item_subst => {
                        let item_substs = ty::ItemSubsts {
                            substs: val_dsr.read_substs(xcx)
                        };
                        dcx.tcx.item_substs.borrow_mut().insert(
                            id, item_substs);
                    }
                    c::tag_table_freevars => {
                        let fv_info = val_dsr.read_to_vec(|val_dsr| {
                            Ok(val_dsr.read_freevar_entry(xcx))
                        }).unwrap().move_iter().collect();
                        dcx.tcx.freevars.borrow_mut().insert(id, fv_info);
                    }
                    c::tag_table_tcache => {
                        let pty = val_dsr.read_polytype(xcx);
                        let lid = ast::DefId { krate: ast::LOCAL_CRATE, node: id };
                        dcx.tcx.tcache.borrow_mut().insert(lid, pty);
                    }
                    c::tag_table_param_defs => {
                        let bounds = val_dsr.read_type_param_def(xcx);
                        dcx.tcx.ty_param_defs.borrow_mut().insert(id, bounds);
                    }
                    c::tag_table_method_map => {
                        let (adjustment, method) = val_dsr.read_method_callee(xcx);
                        let method_call = MethodCall {
                            expr_id: id,
                            adjustment: adjustment
                        };
                        dcx.tcx.method_map.borrow_mut().insert(method_call, method);
                    }
                    c::tag_table_vtable_map => {
                        let (adjustment, vtable_res) =
                            val_dsr.read_vtable_res_with_key(xcx.dcx.tcx,
                                                             xcx.dcx.cdata);
                        let vtable_key = MethodCall {
                            expr_id: id,
                            adjustment: adjustment
                        };
                        dcx.tcx.vtable_map.borrow_mut().insert(vtable_key, vtable_res);
                    }
                    c::tag_table_adjustments => {
                        let adj: ty::AutoAdjustment = val_dsr.read_auto_adjustment(xcx);
                        dcx.tcx.adjustments.borrow_mut().insert(id, adj);
                    }
                    c::tag_table_unboxed_closure_type => {
                        let unboxed_closure_type =
                            val_dsr.read_unboxed_closure_type(xcx);
                        dcx.tcx
                           .unboxed_closure_types
                           .borrow_mut()
                           .insert(ast_util::local_def(id),
                                   unboxed_closure_type);
                    }
                    _ => {
                        xcx.dcx.tcx.sess.bug(
                            format!("unknown tag found in side tables: {:x}",
                                    tag).as_slice());
                    }
                }
            }
        }

        debug!(">< Side table doc loaded");
        true
    });
}

// ______________________________________________________________________
// Testing of astencode_gen

#[cfg(test)]
fn encode_item_ast(ebml_w: &mut Encoder, item: Gc<ast::Item>) {
    ebml_w.start_tag(c::tag_tree as uint);
    (*item).encode(ebml_w);
    ebml_w.end_tag();
}

#[cfg(test)]
fn decode_item_ast(par_doc: ebml::Doc) -> Gc<ast::Item> {
    let chi_doc = par_doc.get(c::tag_tree as uint);
    let mut d = reader::Decoder::new(chi_doc);
    box(GC) Decodable::decode(&mut d).unwrap()
}

#[cfg(test)]
trait fake_ext_ctxt {
    fn cfg(&self) -> ast::CrateConfig;
    fn parse_sess<'a>(&'a self) -> &'a parse::ParseSess;
    fn call_site(&self) -> Span;
    fn ident_of(&self, st: &str) -> ast::Ident;
}

#[cfg(test)]
impl fake_ext_ctxt for parse::ParseSess {
    fn cfg(&self) -> ast::CrateConfig {
        Vec::new()
    }
    fn parse_sess<'a>(&'a self) -> &'a parse::ParseSess { self }
    fn call_site(&self) -> Span {
        codemap::Span {
            lo: codemap::BytePos(0),
            hi: codemap::BytePos(0),
            expn_info: None
        }
    }
    fn ident_of(&self, st: &str) -> ast::Ident {
        token::str_to_ident(st)
    }
}

#[cfg(test)]
fn mk_ctxt() -> parse::ParseSess {
    parse::new_parse_sess()
}

#[cfg(test)]
fn roundtrip(in_item: Option<Gc<ast::Item>>) {
    use std::io::MemWriter;

    let in_item = in_item.unwrap();
    let mut wr = MemWriter::new();
    {
        let mut ebml_w = writer::Encoder::new(&mut wr);
        encode_item_ast(&mut ebml_w, in_item);
    }
    let ebml_doc = ebml::Doc::new(wr.get_ref());
    let out_item = decode_item_ast(ebml_doc);

    assert!(in_item == out_item);
}

#[test]
fn test_basic() {
    let cx = mk_ctxt();
    roundtrip(quote_item!(cx,
        fn foo() {}
    ));
}
/* NOTE: When there's a snapshot, update this (yay quasiquoter!)
#[test]
fn test_smalltalk() {
    let cx = mk_ctxt();
    roundtrip(quote_item!(cx,
        fn foo() -> int { 3 + 4 } // first smalltalk program ever executed.
    ));
}
*/

#[test]
fn test_more() {
    let cx = mk_ctxt();
    roundtrip(quote_item!(cx,
        fn foo(x: uint, y: uint) -> uint {
            let z = x + y;
            return z;
        }
    ));
}

#[test]
fn test_simplification() {
    let cx = mk_ctxt();
    let item = quote_item!(&cx,
        fn new_int_alist<B>() -> alist<int, B> {
            fn eq_int(a: int, b: int) -> bool { a == b }
            return alist {eq_fn: eq_int, data: Vec::new()};
        }
    ).unwrap();
    let item_in = e::IIItemRef(&*item);
    let item_out = simplify_ast(item_in);
    let item_exp = ast::IIItem(quote_item!(cx,
        fn new_int_alist<B>() -> alist<int, B> {
            return alist {eq_fn: eq_int, data: Vec::new()};
        }
    ).unwrap());
    match (item_out, item_exp) {
      (ast::IIItem(item_out), ast::IIItem(item_exp)) => {
        assert!(pprust::item_to_string(&*item_out) ==
                pprust::item_to_string(&*item_exp));
      }
      _ => fail!()
    }
}
