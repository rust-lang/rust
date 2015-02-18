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

use metadata::common as c;
use metadata::cstore as cstore;
use session::Session;
use metadata::decoder;
use middle::def;
use metadata::encoder as e;
use middle::region;
use metadata::tydecode;
use metadata::tydecode::{DefIdSource, NominalType, TypeWithId, TypeParameter};
use metadata::tydecode::{RegionParameter, ClosureSource};
use metadata::tyencode;
use middle::check_const::ConstQualif;
use middle::mem_categorization::Typer;
use middle::subst;
use middle::subst::VecPerParamSpace;
use middle::ty::{self, Ty, MethodCall, MethodCallee, MethodOrigin};
use util::ppaux::ty_to_string;

use syntax::{ast, ast_map, ast_util, codemap, fold};
use syntax::ast_util::PostExpansionMethod;
use syntax::codemap::Span;
use syntax::fold::Folder;
use syntax::parse::token;
use syntax::ptr::P;
use syntax;

use std::old_io::Seek;
use std::num::FromPrimitive;
use std::rc::Rc;

use rbml::io::SeekableMemWriter;
use rbml::{reader, writer};
use rbml;
use serialize;
use serialize::{Decodable, Decoder, DecoderHelpers, Encodable};
use serialize::{EncoderHelpers};

#[cfg(test)] use syntax::parse;
#[cfg(test)] use syntax::print::pprust;

struct DecodeContext<'a, 'b, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    cdata: &'b cstore::crate_metadata,
    from_id_range: ast_util::IdRange,
    to_id_range: ast_util::IdRange
}

trait tr {
    fn tr(&self, dcx: &DecodeContext) -> Self;
}

trait tr_intern {
    fn tr_intern(&self, dcx: &DecodeContext) -> ast::DefId;
}

pub type Encoder<'a> = writer::Encoder<'a, SeekableMemWriter>;

// ______________________________________________________________________
// Top-level methods.

pub fn encode_inlined_item(ecx: &e::EncodeContext,
                           rbml_w: &mut Encoder,
                           ii: e::InlinedItemRef) {
    let id = match ii {
        e::IIItemRef(i) => i.id,
        e::IIForeignRef(i) => i.id,
        e::IITraitItemRef(_, &ast::ProvidedMethod(ref m)) => m.id,
        e::IITraitItemRef(_, &ast::RequiredMethod(ref m)) => m.id,
        e::IITraitItemRef(_, &ast::TypeTraitItem(ref ti)) => ti.ty_param.id,
        e::IIImplItemRef(_, &ast::MethodImplItem(ref m)) => m.id,
        e::IIImplItemRef(_, &ast::TypeImplItem(ref ti)) => ti.id,
    };
    debug!("> Encoding inlined item: {} ({:?})",
           ecx.tcx.map.path_to_string(id),
           rbml_w.writer.tell());

    // Folding could be avoided with a smarter encoder.
    let ii = simplify_ast(ii);
    let id_range = ast_util::compute_id_range_for_inlined_item(&ii);

    rbml_w.start_tag(c::tag_ast as uint);
    id_range.encode(rbml_w);
    encode_ast(rbml_w, &ii);
    encode_side_tables_for_ii(ecx, rbml_w, &ii);
    rbml_w.end_tag();

    debug!("< Encoded inlined fn: {} ({:?})",
           ecx.tcx.map.path_to_string(id),
           rbml_w.writer.tell());
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
    fn new_def_id(&self, def_id: ast::DefId) -> ast::DefId {
        self.tr_def_id(def_id)
    }
    fn new_span(&self, span: Span) -> Span {
        self.tr_span(span)
    }
}

pub fn decode_inlined_item<'tcx>(cdata: &cstore::crate_metadata,
                                 tcx: &ty::ctxt<'tcx>,
                                 path: Vec<ast_map::PathElem>,
                                 par_doc: rbml::Doc)
                                 -> Result<&'tcx ast::InlinedItem, Vec<ast_map::PathElem>> {
    match par_doc.opt_child(c::tag_ast) {
      None => Err(path),
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
            to_id_range: to_id_range
        };
        let raw_ii = decode_ast(ast_doc);
        let ii = ast_map::map_decoded_item(&dcx.tcx.map, path, raw_ii, dcx);

        let ident = match *ii {
            ast::IIItem(ref i) => i.ident,
            ast::IIForeign(ref i) => i.ident,
            ast::IITraitItem(_, ref ti) => {
                match *ti {
                    ast::ProvidedMethod(ref m) => m.pe_ident(),
                    ast::RequiredMethod(ref ty_m) => ty_m.ident,
                    ast::TypeTraitItem(ref ti) => ti.ty_param.ident,
                }
            },
            ast::IIImplItem(_, ref m) => {
                match *m {
                    ast::MethodImplItem(ref m) => m.pe_ident(),
                    ast::TypeImplItem(ref ti) => ti.ident,
                }
            }
        };
        debug!("Fn named: {}", token::get_ident(ident));
        debug!("< Decoded inlined fn: {}::{}",
               path_as_str.unwrap(),
               token::get_ident(ident));
        region::resolve_inlined_item(&tcx.sess, &tcx.region_maps, ii);
        decode_side_tables(dcx, ast_doc);
        match *ii {
          ast::IIItem(ref i) => {
            debug!(">>> DECODED ITEM >>>\n{}\n<<< DECODED ITEM <<<",
                   syntax::print::pprust::item_to_string(&**i));
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
        (id - self.from_id_range.min + self.to_id_range.min)
    }

    /// Translates an EXTERNAL def-id, converting the crate number from the one used in the encoded
    /// data to the current crate numbers..  By external, I mean that it be translated to a
    /// reference to the item in its original crate, as opposed to being translated to a reference
    /// to the inlined version of the item.  This is typically, but not always, what you want,
    /// because most def-ids refer to external things like types or other fns that may or may not
    /// be inlined.  Note that even when the inlined function is referencing itself recursively, we
    /// would want `tr_def_id` for that reference--- conceptually the function calls the original,
    /// non-inlined version, and trans deals with linking that recursive call to the inlined copy.
    ///
    /// However, there are a *few* cases where def-ids are used but we know that the thing being
    /// referenced is in fact *internal* to the item being inlined.  In those cases, you should use
    /// `tr_intern_def_id()` below.
    pub fn tr_def_id(&self, did: ast::DefId) -> ast::DefId {

        decoder::translate_def_id(self.cdata, did)
    }

    /// Translates an INTERNAL def-id, meaning a def-id that is
    /// known to refer to some part of the item currently being
    /// inlined.  In that case, we want to convert the def-id to
    /// refer to the current crate and to the new, inlined node-id.
    pub fn tr_intern_def_id(&self, did: ast::DefId) -> ast::DefId {
        assert_eq!(did.krate, ast::LOCAL_CRATE);
        ast::DefId { krate: ast::LOCAL_CRATE, node: self.tr_id(did.node) }
    }
    pub fn tr_span(&self, _span: Span) -> Span {
        codemap::DUMMY_SP // FIXME (#1972): handle span properly
    }
}

impl tr_intern for ast::DefId {
    fn tr_intern(&self, dcx: &DecodeContext) -> ast::DefId {
        dcx.tr_intern_def_id(*self)
    }
}

impl tr for ast::DefId {
    fn tr(&self, dcx: &DecodeContext) -> ast::DefId {
        dcx.tr_def_id(*self)
    }
}

impl tr for Option<ast::DefId> {
    fn tr(&self, dcx: &DecodeContext) -> Option<ast::DefId> {
        self.map(|d| dcx.tr_def_id(d))
    }
}

impl tr for Span {
    fn tr(&self, dcx: &DecodeContext) -> Span {
        dcx.tr_span(*self)
    }
}

trait def_id_encoder_helpers {
    fn emit_def_id(&mut self, did: ast::DefId);
}

impl<S:serialize::Encoder> def_id_encoder_helpers for S {
    fn emit_def_id(&mut self, did: ast::DefId) {
        did.encode(self).ok().unwrap()
    }
}

trait def_id_decoder_helpers {
    fn read_def_id(&mut self, dcx: &DecodeContext) -> ast::DefId;
    fn read_def_id_nodcx(&mut self,
                         cdata: &cstore::crate_metadata) -> ast::DefId;
}

impl<D:serialize::Decoder> def_id_decoder_helpers for D {
    fn read_def_id(&mut self, dcx: &DecodeContext) -> ast::DefId {
        let did: ast::DefId = Decodable::decode(self).ok().unwrap();
        did.tr(dcx)
    }

    fn read_def_id_nodcx(&mut self,
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

fn encode_ast(rbml_w: &mut Encoder, item: &ast::InlinedItem) {
    rbml_w.start_tag(c::tag_tree as uint);
    item.encode(rbml_w);
    rbml_w.end_tag();
}

struct NestedItemsDropper;

impl Folder for NestedItemsDropper {
    fn fold_block(&mut self, blk: P<ast::Block>) -> P<ast::Block> {
        blk.and_then(|ast::Block {id, stmts, expr, rules, span, ..}| {
            let stmts_sans_items = stmts.into_iter().filter_map(|stmt| {
                let use_stmt = match stmt.node {
                    ast::StmtExpr(_, _) | ast::StmtSemi(_, _) => true,
                    ast::StmtDecl(ref decl, _) => {
                        match decl.node {
                            ast::DeclLocal(_) => true,
                            ast::DeclItem(_) => false,
                        }
                    }
                    ast::StmtMac(..) => panic!("unexpanded macro in astencode")
                };
                if use_stmt {
                    Some(stmt)
                } else {
                    None
                }
            }).collect();
            let blk_sans_items = P(ast::Block {
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
fn simplify_ast(ii: e::InlinedItemRef) -> ast::InlinedItem {
    let mut fld = NestedItemsDropper;

    match ii {
        // HACK we're not dropping items.
        e::IIItemRef(i) => {
            ast::IIItem(fold::noop_fold_item(P(i.clone()), &mut fld)
                            .expect_one("expected one item"))
        }
        e::IITraitItemRef(d, ti) => {
            ast::IITraitItem(d, match *ti {
                ast::ProvidedMethod(ref m) => {
                    ast::ProvidedMethod(
                        fold::noop_fold_method(m.clone(), &mut fld)
                            .expect_one("noop_fold_method must produce \
                                         exactly one method"))
                }
                ast::RequiredMethod(ref ty_m) => {
                    ast::RequiredMethod(
                        fold::noop_fold_type_method(ty_m.clone(), &mut fld))
                }
                ast::TypeTraitItem(ref associated_type) => {
                    ast::TypeTraitItem(
                        P(fold::noop_fold_associated_type(
                            (**associated_type).clone(),
                            &mut fld)))
                }
            })
        }
        e::IIImplItemRef(d, m) => {
            ast::IIImplItem(d, match *m {
                ast::MethodImplItem(ref m) => {
                    ast::MethodImplItem(
                        fold::noop_fold_method(m.clone(), &mut fld)
                            .expect_one("noop_fold_method must produce \
                                         exactly one method"))
                }
                ast::TypeImplItem(ref td) => {
                    ast::TypeImplItem(
                        P(fold::noop_fold_typedef((**td).clone(), &mut fld)))
                }
            })
        }
        e::IIForeignRef(i) => {
            ast::IIForeign(fold::noop_fold_foreign_item(P(i.clone()), &mut fld))
        }
    }
}

fn decode_ast(par_doc: rbml::Doc) -> ast::InlinedItem {
    let chi_doc = par_doc.get(c::tag_tree as uint);
    let mut d = reader::Decoder::new(chi_doc);
    Decodable::decode(&mut d).unwrap()
}

// ______________________________________________________________________
// Encoding and decoding of ast::def

fn decode_def(dcx: &DecodeContext, doc: rbml::Doc) -> def::Def {
    let mut dsr = reader::Decoder::new(doc);
    let def: def::Def = Decodable::decode(&mut dsr).unwrap();
    def.tr(dcx)
}

impl tr for def::Def {
    fn tr(&self, dcx: &DecodeContext) -> def::Def {
        match *self {
          def::DefFn(did, is_ctor) => def::DefFn(did.tr(dcx), is_ctor),
          def::DefStaticMethod(did, p) => {
            def::DefStaticMethod(did.tr(dcx), p.map(|did2| did2.tr(dcx)))
          }
          def::DefMethod(did0, did1, p) => {
            def::DefMethod(did0.tr(dcx),
                           did1.map(|did1| did1.tr(dcx)),
                           p.map(|did2| did2.tr(dcx)))
          }
          def::DefSelfTy(nid) => { def::DefSelfTy(dcx.tr_id(nid)) }
          def::DefMod(did) => { def::DefMod(did.tr(dcx)) }
          def::DefForeignMod(did) => { def::DefForeignMod(did.tr(dcx)) }
          def::DefStatic(did, m) => { def::DefStatic(did.tr(dcx), m) }
          def::DefConst(did) => { def::DefConst(did.tr(dcx)) }
          def::DefLocal(nid) => { def::DefLocal(dcx.tr_id(nid)) }
          def::DefVariant(e_did, v_did, is_s) => {
            def::DefVariant(e_did.tr(dcx), v_did.tr(dcx), is_s)
          },
          def::DefTrait(did) => def::DefTrait(did.tr(dcx)),
          def::DefTy(did, is_enum) => def::DefTy(did.tr(dcx), is_enum),
          def::DefAssociatedTy(did) => def::DefAssociatedTy(did.tr(dcx)),
          def::DefAssociatedPath(def::TyParamProvenance::FromSelf(did), ident) =>
              def::DefAssociatedPath(def::TyParamProvenance::FromSelf(did.tr(dcx)), ident),
          def::DefAssociatedPath(def::TyParamProvenance::FromParam(did), ident) =>
              def::DefAssociatedPath(def::TyParamProvenance::FromParam(did.tr(dcx)), ident),
          def::DefPrimTy(p) => def::DefPrimTy(p),
          def::DefTyParam(s, index, def_id, n) => def::DefTyParam(s, index, def_id.tr(dcx), n),
          def::DefUse(did) => def::DefUse(did.tr(dcx)),
          def::DefUpvar(nid1, nid2) => {
            def::DefUpvar(dcx.tr_id(nid1), dcx.tr_id(nid2))
          }
          def::DefStruct(did) => def::DefStruct(did.tr(dcx)),
          def::DefRegion(nid) => def::DefRegion(dcx.tr_id(nid)),
          def::DefTyParamBinder(nid) => {
            def::DefTyParamBinder(dcx.tr_id(nid))
          }
          def::DefLabel(nid) => def::DefLabel(dcx.tr_id(nid))
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of ancillary information

impl tr for ty::Region {
    fn tr(&self, dcx: &DecodeContext) -> ty::Region {
        match *self {
            ty::ReLateBound(debruijn, br) => {
                ty::ReLateBound(debruijn, br.tr(dcx))
            }
            ty::ReEarlyBound(id, space, index, ident) => {
                ty::ReEarlyBound(dcx.tr_id(id), space, index, ident)
            }
            ty::ReScope(scope) => {
                ty::ReScope(scope.tr(dcx))
            }
            ty::ReEmpty | ty::ReStatic | ty::ReInfer(..) => {
                *self
            }
            ty::ReFree(ref fr) => {
                ty::ReFree(fr.tr(dcx))
            }
        }
    }
}

impl tr for ty::FreeRegion {
    fn tr(&self, dcx: &DecodeContext) -> ty::FreeRegion {
        ty::FreeRegion { scope: self.scope.tr(dcx),
                         bound_region: self.bound_region.tr(dcx) }
    }
}

impl tr for region::CodeExtent {
    fn tr(&self, dcx: &DecodeContext) -> region::CodeExtent {
        self.map_id(|id| dcx.tr_id(id))
    }
}

impl tr for region::DestructionScopeData {
    fn tr(&self, dcx: &DecodeContext) -> region::DestructionScopeData {
        region::DestructionScopeData { node_id: dcx.tr_id(self.node_id) }
    }
}

impl tr for ty::BoundRegion {
    fn tr(&self, dcx: &DecodeContext) -> ty::BoundRegion {
        match *self {
            ty::BrAnon(_) |
            ty::BrFresh(_) |
            ty::BrEnv => *self,
            ty::BrNamed(id, ident) => ty::BrNamed(dcx.tr_def_id(id),
                                                    ident),
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
    fn read_capture_mode(&mut self) -> ast::CaptureClause;
}

impl<'a> rbml_decoder_helper for reader::Decoder<'a> {
    fn read_freevar_entry(&mut self, dcx: &DecodeContext)
                          -> ty::Freevar {
        let fv: ty::Freevar = Decodable::decode(self).unwrap();
        fv.tr(dcx)
    }

    fn read_capture_mode(&mut self) -> ast::CaptureClause {
        let cm: ast::CaptureClause = Decodable::decode(self).unwrap();
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

impl tr for ty::UpvarBorrow {
    fn tr(&self, dcx: &DecodeContext) -> ty::UpvarBorrow {
        ty::UpvarBorrow {
            kind: self.kind,
            region: self.region.tr(dcx)
        }
    }
}

impl tr for ty::UpvarCapture {
    fn tr(&self, dcx: &DecodeContext) -> ty::UpvarCapture {
        match *self {
            ty::UpvarCapture::ByValue => ty::UpvarCapture::ByValue,
            ty::UpvarCapture::ByRef(ref data) => ty::UpvarCapture::ByRef(data.tr(dcx)),
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of MethodCallee

trait read_method_callee_helper<'tcx> {
    fn read_method_callee<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
        -> (ty::ExprAdjustment, MethodCallee<'tcx>);
}

fn encode_method_callee<'a, 'tcx>(ecx: &e::EncodeContext<'a, 'tcx>,
                                  rbml_w: &mut Encoder,
                                  adjustment: ty::ExprAdjustment,
                                  method: &MethodCallee<'tcx>) {
    use serialize::Encoder;

    rbml_w.emit_struct("MethodCallee", 4, |rbml_w| {
        rbml_w.emit_struct_field("adjustment", 0, |rbml_w| {
            adjustment.encode(rbml_w)
        });
        rbml_w.emit_struct_field("origin", 1, |rbml_w| {
            Ok(rbml_w.emit_method_origin(ecx, &method.origin))
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
        -> (ty::ExprAdjustment, MethodCallee<'tcx>) {

        self.read_struct("MethodCallee", 4, |this| {
            let adjustment = this.read_struct_field("adjustment", 0, |this| {
                Decodable::decode(this)
            }).unwrap();
            Ok((adjustment, MethodCallee {
                origin: this.read_struct_field("origin", 1, |this| {
                    Ok(this.read_method_origin(dcx))
                }).unwrap(),
                ty: this.read_struct_field("ty", 2, |this| {
                    Ok(this.read_ty(dcx))
                }).unwrap(),
                substs: this.read_struct_field("substs", 3, |this| {
                    Ok(this.read_substs(dcx))
                }).unwrap()
            }))
        }).unwrap()
    }
}

impl<'tcx> tr for MethodOrigin<'tcx> {
    fn tr(&self, dcx: &DecodeContext) -> MethodOrigin<'tcx> {
        match *self {
            ty::MethodStatic(did) => ty::MethodStatic(did.tr(dcx)),
            ty::MethodStaticClosure(did) => {
                ty::MethodStaticClosure(did.tr(dcx))
            }
            ty::MethodTypeParam(ref mp) => {
                ty::MethodTypeParam(
                    ty::MethodParam {
                        // def-id is already translated when we read it out
                        trait_ref: mp.trait_ref.clone(),
                        method_num: mp.method_num,
                        impl_def_id: mp.impl_def_id.tr(dcx),
                    }
                )
            }
            ty::MethodTraitObject(ref mo) => {
                ty::MethodTraitObject(
                    ty::MethodObject {
                        trait_ref: mo.trait_ref.clone(),
                        .. *mo
                    }
                )
            }
        }
    }
}

pub fn encode_closure_kind(ebml_w: &mut Encoder, kind: ty::ClosureKind) {
    kind.encode(ebml_w).unwrap();
}

pub trait vtable_decoder_helpers<'tcx> {
    fn read_vec_per_param_space<T, F>(&mut self, f: F) -> VecPerParamSpace<T> where
        F: FnMut(&mut Self) -> T;
    fn read_vtable_res_with_key(&mut self,
                                tcx: &ty::ctxt<'tcx>,
                                cdata: &cstore::crate_metadata)
                                -> (ty::ExprAdjustment, ty::vtable_res<'tcx>);
    fn read_vtable_res(&mut self,
                       tcx: &ty::ctxt<'tcx>, cdata: &cstore::crate_metadata)
                      -> ty::vtable_res<'tcx>;
    fn read_vtable_param_res(&mut self,
                       tcx: &ty::ctxt<'tcx>, cdata: &cstore::crate_metadata)
                      -> ty::vtable_param_res<'tcx>;
    fn read_vtable_origin(&mut self,
                          tcx: &ty::ctxt<'tcx>, cdata: &cstore::crate_metadata)
                          -> ty::vtable_origin<'tcx>;
}

impl<'tcx, 'a> vtable_decoder_helpers<'tcx> for reader::Decoder<'a> {
    fn read_vec_per_param_space<T, F>(&mut self, mut f: F) -> VecPerParamSpace<T> where
        F: FnMut(&mut reader::Decoder<'a>) -> T,
    {
        let types = self.read_to_vec(|this| Ok(f(this))).unwrap();
        let selfs = self.read_to_vec(|this| Ok(f(this))).unwrap();
        let fns = self.read_to_vec(|this| Ok(f(this))).unwrap();
        VecPerParamSpace::new(types, selfs, fns)
    }

    fn read_vtable_res_with_key(&mut self,
                                tcx: &ty::ctxt<'tcx>,
                                cdata: &cstore::crate_metadata)
                                -> (ty::ExprAdjustment, ty::vtable_res<'tcx>) {
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
                       tcx: &ty::ctxt<'tcx>,
                       cdata: &cstore::crate_metadata)
                       -> ty::vtable_res<'tcx>
    {
        self.read_vec_per_param_space(
            |this| this.read_vtable_param_res(tcx, cdata))
    }

    fn read_vtable_param_res(&mut self,
                             tcx: &ty::ctxt<'tcx>, cdata: &cstore::crate_metadata)
                      -> ty::vtable_param_res<'tcx> {
        self.read_to_vec(|this| Ok(this.read_vtable_origin(tcx, cdata)))
             .unwrap().into_iter().collect()
    }

    fn read_vtable_origin(&mut self,
                          tcx: &ty::ctxt<'tcx>, cdata: &cstore::crate_metadata)
        -> ty::vtable_origin<'tcx> {
        self.read_enum("vtable_origin", |this| {
            this.read_enum_variant(&["vtable_static",
                                     "vtable_param",
                                     "vtable_error",
                                     "vtable_closure"],
                                   |this, i| {
                Ok(match i {
                  0 => {
                    ty::vtable_static(
                        this.read_enum_variant_arg(0, |this| {
                            Ok(this.read_def_id_nodcx(cdata))
                        }).unwrap(),
                        this.read_enum_variant_arg(1, |this| {
                            Ok(this.read_substs_nodcx(tcx, cdata))
                        }).unwrap(),
                        this.read_enum_variant_arg(2, |this| {
                            Ok(this.read_vtable_res(tcx, cdata))
                        }).unwrap()
                    )
                  }
                  1 => {
                    ty::vtable_param(
                        this.read_enum_variant_arg(0, |this| {
                            Decodable::decode(this)
                        }).unwrap(),
                        this.read_enum_variant_arg(1, |this| {
                            this.read_uint()
                        }).unwrap()
                    )
                  }
                  2 => {
                    ty::vtable_closure(
                        this.read_enum_variant_arg(0, |this| {
                            Ok(this.read_def_id_nodcx(cdata))
                        }).unwrap()
                    )
                  }
                  3 => {
                    ty::vtable_error
                  }
                  _ => panic!("bad enum variant")
                })
            })
        }).unwrap()
    }
}

// ___________________________________________________________________________
//

fn encode_vec_per_param_space<T, F>(rbml_w: &mut Encoder,
                                    v: &subst::VecPerParamSpace<T>,
                                    mut f: F) where
    F: FnMut(&mut Encoder, &T),
{
    for &space in &subst::ParamSpace::all() {
        rbml_w.emit_from_vec(v.get_slice(space),
                             |rbml_w, n| Ok(f(rbml_w, n))).unwrap();
    }
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
    fn emit_closure_type<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                             closure_type: &ty::ClosureTy<'tcx>);
    fn emit_method_origin<'a>(&mut self,
                              ecx: &e::EncodeContext<'a, 'tcx>,
                              method_origin: &ty::MethodOrigin<'tcx>);
    fn emit_ty<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>, ty: Ty<'tcx>);
    fn emit_tys<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>, tys: &[Ty<'tcx>]);
    fn emit_type_param_def<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                               type_param_def: &ty::TypeParameterDef<'tcx>);
    fn emit_predicate<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                          predicate: &ty::Predicate<'tcx>);
    fn emit_trait_ref<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                          ty: &ty::TraitRef<'tcx>);
    fn emit_type_scheme<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                            type_scheme: ty::TypeScheme<'tcx>);
    fn emit_substs<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                       substs: &subst::Substs<'tcx>);
    fn emit_existential_bounds<'b>(&mut self, ecx: &e::EncodeContext<'b,'tcx>,
                                   bounds: &ty::ExistentialBounds<'tcx>);
    fn emit_builtin_bounds(&mut self, ecx: &e::EncodeContext, bounds: &ty::BuiltinBounds);
    fn emit_auto_adjustment<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                                adj: &ty::AutoAdjustment<'tcx>);
    fn emit_autoref<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                        autoref: &ty::AutoRef<'tcx>);
    fn emit_auto_deref_ref<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                               auto_deref_ref: &ty::AutoDerefRef<'tcx>);
    fn emit_unsize_kind<'a>(&mut self, ecx: &e::EncodeContext<'a, 'tcx>,
                            uk: &ty::UnsizeKind<'tcx>);
}

impl<'a, 'tcx> rbml_writer_helpers<'tcx> for Encoder<'a> {
    fn emit_closure_type<'b>(&mut self,
                             ecx: &e::EncodeContext<'b, 'tcx>,
                             closure_type: &ty::ClosureTy<'tcx>) {
        self.emit_opaque(|this| {
            Ok(e::write_closure_type(ecx, this, closure_type))
        });
    }

    fn emit_method_origin<'b>(&mut self,
                              ecx: &e::EncodeContext<'b, 'tcx>,
                              method_origin: &ty::MethodOrigin<'tcx>)
    {
        use serialize::Encoder;

        self.emit_enum("MethodOrigin", |this| {
            match *method_origin {
                ty::MethodStatic(def_id) => {
                    this.emit_enum_variant("MethodStatic", 0, 1, |this| {
                        Ok(this.emit_def_id(def_id))
                    })
                }

                ty::MethodStaticClosure(def_id) => {
                    this.emit_enum_variant("MethodStaticClosure", 1, 1, |this| {
                        Ok(this.emit_def_id(def_id))
                    })
                }

                ty::MethodTypeParam(ref p) => {
                    this.emit_enum_variant("MethodTypeParam", 2, 1, |this| {
                        this.emit_struct("MethodParam", 2, |this| {
                            try!(this.emit_struct_field("trait_ref", 0, |this| {
                                Ok(this.emit_trait_ref(ecx, &*p.trait_ref))
                            }));
                            try!(this.emit_struct_field("method_num", 0, |this| {
                                this.emit_uint(p.method_num)
                            }));
                            try!(this.emit_struct_field("impl_def_id", 0, |this| {
                                this.emit_option(|this| {
                                    match p.impl_def_id {
                                        None => this.emit_option_none(),
                                        Some(did) => this.emit_option_some(|this| {
                                            Ok(this.emit_def_id(did))
                                        })
                                    }
                                })
                            }));
                            Ok(())
                        })
                    })
                }

                ty::MethodTraitObject(ref o) => {
                    this.emit_enum_variant("MethodTraitObject", 3, 1, |this| {
                        this.emit_struct("MethodObject", 2, |this| {
                            try!(this.emit_struct_field("trait_ref", 0, |this| {
                                Ok(this.emit_trait_ref(ecx, &*o.trait_ref))
                            }));
                            try!(this.emit_struct_field("object_trait_id", 0, |this| {
                                Ok(this.emit_def_id(o.object_trait_id))
                            }));
                            try!(this.emit_struct_field("method_num", 0, |this| {
                                this.emit_uint(o.method_num)
                            }));
                            try!(this.emit_struct_field("vtable_index", 0, |this| {
                                this.emit_uint(o.vtable_index)
                            }));
                            Ok(())
                        })
                    })
                }
            }
        });
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

    fn emit_type_param_def<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>,
                               type_param_def: &ty::TypeParameterDef<'tcx>) {
        self.emit_opaque(|this| {
            Ok(tyencode::enc_type_param_def(this.writer,
                                         &ecx.ty_str_ctxt(),
                                         type_param_def))
        });
    }

    fn emit_predicate<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>,
                          predicate: &ty::Predicate<'tcx>) {
        self.emit_opaque(|this| {
            Ok(tyencode::enc_predicate(this.writer,
                                       &ecx.ty_str_ctxt(),
                                       predicate))
        });
    }

    fn emit_type_scheme<'b>(&mut self,
                            ecx: &e::EncodeContext<'b, 'tcx>,
                            type_scheme: ty::TypeScheme<'tcx>) {
        use serialize::Encoder;

        self.emit_struct("TypeScheme", 2, |this| {
            this.emit_struct_field("generics", 0, |this| {
                this.emit_struct("Generics", 2, |this| {
                    this.emit_struct_field("types", 0, |this| {
                        Ok(encode_vec_per_param_space(
                            this, &type_scheme.generics.types,
                            |this, def| this.emit_type_param_def(ecx, def)))
                    });
                    this.emit_struct_field("regions", 1, |this| {
                        Ok(encode_vec_per_param_space(
                            this, &type_scheme.generics.regions,
                            |this, def| def.encode(this).unwrap()))
                    })
                })
            });
            this.emit_struct_field("ty", 1, |this| {
                Ok(this.emit_ty(ecx, type_scheme.ty))
            })
        });
    }

    fn emit_existential_bounds<'b>(&mut self, ecx: &e::EncodeContext<'b,'tcx>,
                                   bounds: &ty::ExistentialBounds<'tcx>) {
        self.emit_opaque(|this| Ok(tyencode::enc_existential_bounds(this.writer,
                                                                    &ecx.ty_str_ctxt(),
                                                                    bounds)));
    }

    fn emit_builtin_bounds(&mut self, ecx: &e::EncodeContext, bounds: &ty::BuiltinBounds) {
        self.emit_opaque(|this| Ok(tyencode::enc_builtin_bounds(this.writer,
                                                                &ecx.ty_str_ctxt(),
                                                                bounds)));
    }

    fn emit_substs<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>,
                       substs: &subst::Substs<'tcx>) {
        self.emit_opaque(|this| Ok(tyencode::enc_substs(this.writer,
                                                           &ecx.ty_str_ctxt(),
                                                           substs)));
    }

    fn emit_auto_adjustment<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>,
                                adj: &ty::AutoAdjustment<'tcx>) {
        use serialize::Encoder;

        self.emit_enum("AutoAdjustment", |this| {
            match *adj {
                ty::AdjustReifyFnPointer(def_id) => {
                    this.emit_enum_variant("AdjustReifyFnPointer", 1, 2, |this| {
                        this.emit_enum_variant_arg(0, |this| def_id.encode(this))
                    })
                }

                ty::AdjustDerefRef(ref auto_deref_ref) => {
                    this.emit_enum_variant("AdjustDerefRef", 2, 2, |this| {
                        this.emit_enum_variant_arg(0,
                            |this| Ok(this.emit_auto_deref_ref(ecx, auto_deref_ref)))
                    })
                }
            }
        });
    }

    fn emit_autoref<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>,
                        autoref: &ty::AutoRef<'tcx>) {
        use serialize::Encoder;

        self.emit_enum("AutoRef", |this| {
            match autoref {
                &ty::AutoPtr(r, m, None) => {
                    this.emit_enum_variant("AutoPtr", 0, 3, |this| {
                        this.emit_enum_variant_arg(0, |this| r.encode(this));
                        this.emit_enum_variant_arg(1, |this| m.encode(this));
                        this.emit_enum_variant_arg(2,
                            |this| this.emit_option(|this| this.emit_option_none()))
                    })
                }
                &ty::AutoPtr(r, m, Some(box ref a)) => {
                    this.emit_enum_variant("AutoPtr", 0, 3, |this| {
                        this.emit_enum_variant_arg(0, |this| r.encode(this));
                        this.emit_enum_variant_arg(1, |this| m.encode(this));
                        this.emit_enum_variant_arg(2, |this| this.emit_option(
                            |this| this.emit_option_some(|this| Ok(this.emit_autoref(ecx, a)))))
                    })
                }
                &ty::AutoUnsize(ref uk) => {
                    this.emit_enum_variant("AutoUnsize", 1, 1, |this| {
                        this.emit_enum_variant_arg(0, |this| Ok(this.emit_unsize_kind(ecx, uk)))
                    })
                }
                &ty::AutoUnsizeUniq(ref uk) => {
                    this.emit_enum_variant("AutoUnsizeUniq", 2, 1, |this| {
                        this.emit_enum_variant_arg(0, |this| Ok(this.emit_unsize_kind(ecx, uk)))
                    })
                }
                &ty::AutoUnsafe(m, None) => {
                    this.emit_enum_variant("AutoUnsafe", 3, 2, |this| {
                        this.emit_enum_variant_arg(0, |this| m.encode(this));
                        this.emit_enum_variant_arg(1,
                            |this| this.emit_option(|this| this.emit_option_none()))
                    })
                }
                &ty::AutoUnsafe(m, Some(box ref a)) => {
                    this.emit_enum_variant("AutoUnsafe", 3, 2, |this| {
                        this.emit_enum_variant_arg(0, |this| m.encode(this));
                        this.emit_enum_variant_arg(1, |this| this.emit_option(
                            |this| this.emit_option_some(|this| Ok(this.emit_autoref(ecx, a)))))
                    })
                }
            }
        });
    }

    fn emit_auto_deref_ref<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>,
                               auto_deref_ref: &ty::AutoDerefRef<'tcx>) {
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
            })
        });
    }

    fn emit_unsize_kind<'b>(&mut self, ecx: &e::EncodeContext<'b, 'tcx>,
                            uk: &ty::UnsizeKind<'tcx>) {
        use serialize::Encoder;

        self.emit_enum("UnsizeKind", |this| {
            match *uk {
                ty::UnsizeLength(len) => {
                    this.emit_enum_variant("UnsizeLength", 0, 1, |this| {
                        this.emit_enum_variant_arg(0, |this| len.encode(this))
                    })
                }
                ty::UnsizeStruct(box ref uk, idx) => {
                    this.emit_enum_variant("UnsizeStruct", 1, 2, |this| {
                        this.emit_enum_variant_arg(0, |this| Ok(this.emit_unsize_kind(ecx, uk)));
                        this.emit_enum_variant_arg(1, |this| idx.encode(this))
                    })
                }
                ty::UnsizeVtable(ty::TyTrait { ref principal,
                                               bounds: ref b },
                                 self_ty) => {
                    this.emit_enum_variant("UnsizeVtable", 2, 4, |this| {
                        this.emit_enum_variant_arg(0, |this| {
                            try!(this.emit_struct_field("principal", 0, |this| {
                                Ok(this.emit_trait_ref(ecx, &*principal.0))
                            }));
                            this.emit_struct_field("bounds", 1, |this| {
                                Ok(this.emit_existential_bounds(ecx, b))
                            })
                        });
                        this.emit_enum_variant_arg(1, |this| Ok(this.emit_ty(ecx, self_ty)))
                    })
                }
            }
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
        self.start_tag(tag_id as uint);
        f(self);
        self.end_tag();
    }

    fn id(&mut self, id: ast::NodeId) {
        self.wr_tagged_u64(c::tag_table_id as uint, id as u64);
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
                             ii: &ast::InlinedItem) {
    rbml_w.start_tag(c::tag_table as uint);
    ast_util::visit_ids_for_inlined_item(ii, &mut SideTableEncodingIdVisitor {
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

    if let Some(def) = tcx.def_map.borrow().get(&id) {
        rbml_w.tag(c::tag_table_def, |rbml_w| {
            rbml_w.id(id);
            rbml_w.tag(c::tag_table_val, |rbml_w| (*def).encode(rbml_w).unwrap());
        })
    }

    if let Some(ty) = tcx.node_types.borrow().get(&id) {
        rbml_w.tag(c::tag_table_node_type, |rbml_w| {
            rbml_w.id(id);
            rbml_w.tag(c::tag_table_val, |rbml_w| {
                rbml_w.emit_ty(ecx, *ty);
            })
        })
    }

    if let Some(item_substs) = tcx.item_substs.borrow().get(&id) {
        rbml_w.tag(c::tag_table_item_subst, |rbml_w| {
            rbml_w.id(id);
            rbml_w.tag(c::tag_table_val, |rbml_w| {
                rbml_w.emit_substs(ecx, &item_substs.substs);
            })
        })
    }

    if let Some(fv) = tcx.freevars.borrow().get(&id) {
        rbml_w.tag(c::tag_table_freevars, |rbml_w| {
            rbml_w.id(id);
            rbml_w.tag(c::tag_table_val, |rbml_w| {
                rbml_w.emit_from_vec(fv, |rbml_w, fv_entry| {
                    Ok(encode_freevar_entry(rbml_w, fv_entry))
                });
            })
        });

        for freevar in fv {
            rbml_w.tag(c::tag_table_upvar_capture_map, |rbml_w| {
                rbml_w.id(id);
                rbml_w.tag(c::tag_table_val, |rbml_w| {
                    let var_id = freevar.def.def_id().node;
                    let upvar_id = ty::UpvarId {
                        var_id: var_id,
                        closure_expr_id: id
                    };
                    let upvar_capture = tcx.upvar_capture_map.borrow()[upvar_id].clone();
                    var_id.encode(rbml_w);
                    upvar_capture.encode(rbml_w);
                })
            })
        }
    }

    let lid = ast::DefId { krate: ast::LOCAL_CRATE, node: id };
    if let Some(type_scheme) = tcx.tcache.borrow().get(&lid) {
        rbml_w.tag(c::tag_table_tcache, |rbml_w| {
            rbml_w.id(id);
            rbml_w.tag(c::tag_table_val, |rbml_w| {
                rbml_w.emit_type_scheme(ecx, type_scheme.clone());
            })
        })
    }

    if let Some(type_param_def) = tcx.ty_param_defs.borrow().get(&id) {
        rbml_w.tag(c::tag_table_param_defs, |rbml_w| {
            rbml_w.id(id);
            rbml_w.tag(c::tag_table_val, |rbml_w| {
                rbml_w.emit_type_param_def(ecx, type_param_def)
            })
        })
    }

    let method_call = MethodCall::expr(id);
    if let Some(method) = tcx.method_map.borrow().get(&method_call) {
        rbml_w.tag(c::tag_table_method_map, |rbml_w| {
            rbml_w.id(id);
            rbml_w.tag(c::tag_table_val, |rbml_w| {
                encode_method_callee(ecx, rbml_w, method_call.adjustment, method)
            })
        })
    }

    if let Some(trait_ref) = tcx.object_cast_map.borrow().get(&id) {
        rbml_w.tag(c::tag_table_object_cast_map, |rbml_w| {
            rbml_w.id(id);
            rbml_w.tag(c::tag_table_val, |rbml_w| {
                rbml_w.emit_trait_ref(ecx, &*trait_ref.0);
            })
        })
    }

    if let Some(adjustment) = tcx.adjustments.borrow().get(&id) {
        match *adjustment {
            _ if ty::adjust_is_object(adjustment) => {
                let method_call = MethodCall::autoobject(id);
                if let Some(method) = tcx.method_map.borrow().get(&method_call) {
                    rbml_w.tag(c::tag_table_method_map, |rbml_w| {
                        rbml_w.id(id);
                        rbml_w.tag(c::tag_table_val, |rbml_w| {
                            encode_method_callee(ecx, rbml_w, method_call.adjustment, method)
                        })
                    })
                }
            }
            ty::AdjustDerefRef(ref adj) => {
                assert!(!ty::adjust_is_object(adjustment));
                for autoderef in 0..adj.autoderefs {
                    let method_call = MethodCall::autoderef(id, autoderef);
                    if let Some(method) = tcx.method_map.borrow().get(&method_call) {
                        rbml_w.tag(c::tag_table_method_map, |rbml_w| {
                            rbml_w.id(id);
                            rbml_w.tag(c::tag_table_val, |rbml_w| {
                                encode_method_callee(ecx, rbml_w,
                                                     method_call.adjustment, method)
                            })
                        })
                    }
                }
            }
            _ => {
                assert!(!ty::adjust_is_object(adjustment));
            }
        }

        rbml_w.tag(c::tag_table_adjustments, |rbml_w| {
            rbml_w.id(id);
            rbml_w.tag(c::tag_table_val, |rbml_w| {
                rbml_w.emit_auto_adjustment(ecx, adjustment);
            })
        })
    }

    if let Some(closure_type) = tcx.closure_tys.borrow().get(&ast_util::local_def(id)) {
        rbml_w.tag(c::tag_table_closure_tys, |rbml_w| {
            rbml_w.id(id);
            rbml_w.tag(c::tag_table_val, |rbml_w| {
                rbml_w.emit_closure_type(ecx, closure_type);
            })
        })
    }

    if let Some(closure_kind) = tcx.closure_kinds.borrow().get(&ast_util::local_def(id)) {
        rbml_w.tag(c::tag_table_closure_kinds, |rbml_w| {
            rbml_w.id(id);
            rbml_w.tag(c::tag_table_val, |rbml_w| {
                encode_closure_kind(rbml_w, *closure_kind)
            })
        })
    }

    for &qualif in tcx.const_qualif_map.borrow().get(&id).iter() {
        rbml_w.tag(c::tag_table_const_qualif, |rbml_w| {
            rbml_w.id(id);
            rbml_w.tag(c::tag_table_val, |rbml_w| {
                qualif.encode(rbml_w).unwrap()
            })
        })
    }
}

trait doc_decoder_helpers {
    fn as_int(&self) -> int;
    fn opt_child(&self, tag: c::astencode_tag) -> Option<Self>;
}

impl<'a> doc_decoder_helpers for rbml::Doc<'a> {
    fn as_int(&self) -> int { reader::doc_as_u64(*self) as int }
    fn opt_child(&self, tag: c::astencode_tag) -> Option<rbml::Doc<'a>> {
        reader::maybe_get_doc(*self, tag as uint)
    }
}

trait rbml_decoder_decoder_helpers<'tcx> {
    fn read_method_origin<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                  -> ty::MethodOrigin<'tcx>;
    fn read_ty<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>) -> Ty<'tcx>;
    fn read_tys<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>) -> Vec<Ty<'tcx>>;
    fn read_trait_ref<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                              -> Rc<ty::TraitRef<'tcx>>;
    fn read_poly_trait_ref<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                   -> ty::PolyTraitRef<'tcx>;
    fn read_type_param_def<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                   -> ty::TypeParameterDef<'tcx>;
    fn read_predicate<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                              -> ty::Predicate<'tcx>;
    fn read_type_scheme<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                -> ty::TypeScheme<'tcx>;
    fn read_existential_bounds<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                       -> ty::ExistentialBounds<'tcx>;
    fn read_substs<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                           -> subst::Substs<'tcx>;
    fn read_auto_adjustment<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                    -> ty::AutoAdjustment<'tcx>;
    fn read_closure_kind<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                 -> ty::ClosureKind;
    fn read_closure_ty<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                               -> ty::ClosureTy<'tcx>;
    fn read_auto_deref_ref<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                   -> ty::AutoDerefRef<'tcx>;
    fn read_autoref<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                            -> ty::AutoRef<'tcx>;
    fn read_unsize_kind<'a, 'b>(&mut self, dcx: &DecodeContext<'a, 'b, 'tcx>)
                                -> ty::UnsizeKind<'tcx>;
    fn convert_def_id(&mut self,
                      dcx: &DecodeContext,
                      source: DefIdSource,
                      did: ast::DefId)
                      -> ast::DefId;

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
                     tcx: &ty::ctxt<'tcx>, cdata: &cstore::crate_metadata) -> Ty<'tcx> {
        self.read_opaque(|_, doc| {
            Ok(tydecode::parse_ty_data(
                doc.data,
                cdata.cnum,
                doc.start,
                tcx,
                |_, id| decoder::translate_def_id(cdata, id)))
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
            Ok(tydecode::parse_substs_data(
                doc.data,
                cdata.cnum,
                doc.start,
                tcx,
                |_, id| decoder::translate_def_id(cdata, id)))
        }).unwrap()
    }

    fn read_method_origin<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                                  -> ty::MethodOrigin<'tcx>
    {
        self.read_enum("MethodOrigin", |this| {
            let variants = &["MethodStatic", "MethodStaticClosure",
                             "MethodTypeParam", "MethodTraitObject"];
            this.read_enum_variant(variants, |this, i| {
                Ok(match i {
                    0 => {
                        let def_id = this.read_def_id(dcx);
                        ty::MethodStatic(def_id)
                    }

                    1 => {
                        let def_id = this.read_def_id(dcx);
                        ty::MethodStaticClosure(def_id)
                    }

                    2 => {
                        this.read_struct("MethodTypeParam", 2, |this| {
                            Ok(ty::MethodTypeParam(
                                ty::MethodParam {
                                    trait_ref: {
                                        this.read_struct_field("trait_ref", 0, |this| {
                                            Ok(this.read_trait_ref(dcx))
                                        }).unwrap()
                                    },
                                    method_num: {
                                        this.read_struct_field("method_num", 1, |this| {
                                            this.read_uint()
                                        }).unwrap()
                                    },
                                    impl_def_id: {
                                        this.read_struct_field("impl_def_id", 2, |this| {
                                            this.read_option(|this, b| {
                                                if b {
                                                    Ok(Some(this.read_def_id(dcx)))
                                                } else {
                                                    Ok(None)
                                                }
                                            })
                                        }).unwrap()
                                    }
                                }))
                        }).unwrap()
                    }

                    3 => {
                        this.read_struct("MethodTraitObject", 2, |this| {
                            Ok(ty::MethodTraitObject(
                                ty::MethodObject {
                                    trait_ref: {
                                        this.read_struct_field("trait_ref", 0, |this| {
                                            Ok(this.read_trait_ref(dcx))
                                        }).unwrap()
                                    },
                                    object_trait_id: {
                                        this.read_struct_field("object_trait_id", 1, |this| {
                                            Ok(this.read_def_id(dcx))
                                        }).unwrap()
                                    },
                                    method_num: {
                                        this.read_struct_field("method_num", 2, |this| {
                                            this.read_uint()
                                        }).unwrap()
                                    },
                                    vtable_index: {
                                        this.read_struct_field("vtable_index", 3, |this| {
                                            this.read_uint()
                                        }).unwrap()
                                    },
                                }))
                        }).unwrap()
                    }

                    _ => panic!("..")
                })
            })
        }).unwrap()
    }


    fn read_ty<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>) -> Ty<'tcx> {
        // Note: regions types embed local node ids.  In principle, we
        // should translate these node ids into the new decode
        // context.  However, we do not bother, because region types
        // are not used during trans.

        return self.read_opaque(|this, doc| {
            debug!("read_ty({})", type_string(doc));

            let ty = tydecode::parse_ty_data(
                doc.data,
                dcx.cdata.cnum,
                doc.start,
                dcx.tcx,
                |s, a| this.convert_def_id(dcx, s, a));

            Ok(ty)
        }).unwrap();

        fn type_string(doc: rbml::Doc) -> String {
            let mut str = String::new();
            for i in doc.start..doc.end {
                str.push(doc.data[i] as char);
            }
            str
        }
    }

    fn read_tys<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                        -> Vec<Ty<'tcx>> {
        self.read_to_vec(|this| Ok(this.read_ty(dcx))).unwrap().into_iter().collect()
    }

    fn read_trait_ref<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                              -> Rc<ty::TraitRef<'tcx>> {
        self.read_opaque(|this, doc| {
            let ty = tydecode::parse_trait_ref_data(
                doc.data,
                dcx.cdata.cnum,
                doc.start,
                dcx.tcx,
                |s, a| this.convert_def_id(dcx, s, a));
            Ok(ty)
        }).unwrap()
    }

    fn read_poly_trait_ref<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                                   -> ty::PolyTraitRef<'tcx> {
        ty::Binder(self.read_opaque(|this, doc| {
            let ty = tydecode::parse_trait_ref_data(
                doc.data,
                dcx.cdata.cnum,
                doc.start,
                dcx.tcx,
                |s, a| this.convert_def_id(dcx, s, a));
            Ok(ty)
        }).unwrap())
    }

    fn read_type_param_def<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                                   -> ty::TypeParameterDef<'tcx> {
        self.read_opaque(|this, doc| {
            Ok(tydecode::parse_type_param_def_data(
                doc.data,
                doc.start,
                dcx.cdata.cnum,
                dcx.tcx,
                |s, a| this.convert_def_id(dcx, s, a)))
        }).unwrap()
    }

    fn read_predicate<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                              -> ty::Predicate<'tcx>
    {
        self.read_opaque(|this, doc| {
            Ok(tydecode::parse_predicate_data(doc.data, doc.start, dcx.cdata.cnum, dcx.tcx,
                                              |s, a| this.convert_def_id(dcx, s, a)))
        }).unwrap()
    }

    fn read_type_scheme<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                                -> ty::TypeScheme<'tcx> {
        self.read_struct("TypeScheme", 3, |this| {
            Ok(ty::TypeScheme {
                generics: this.read_struct_field("generics", 0, |this| {
                    this.read_struct("Generics", 2, |this| {
                        Ok(ty::Generics {
                            types:
                            this.read_struct_field("types", 0, |this| {
                                Ok(this.read_vec_per_param_space(
                                    |this| this.read_type_param_def(dcx)))
                            }).unwrap(),

                            regions:
                            this.read_struct_field("regions", 1, |this| {
                                Ok(this.read_vec_per_param_space(
                                    |this| Decodable::decode(this).unwrap()))
                            }).unwrap(),
                        })
                    })
                }).unwrap(),
                ty: this.read_struct_field("ty", 1, |this| {
                    Ok(this.read_ty(dcx))
                }).unwrap()
            })
        }).unwrap()
    }

    fn read_existential_bounds<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                                       -> ty::ExistentialBounds<'tcx>
    {
        self.read_opaque(|this, doc| {
            Ok(tydecode::parse_existential_bounds_data(doc.data,
                                                       dcx.cdata.cnum,
                                                       doc.start,
                                                       dcx.tcx,
                                                       |s, a| this.convert_def_id(dcx, s, a)))
        }).unwrap()
    }

    fn read_substs<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                           -> subst::Substs<'tcx> {
        self.read_opaque(|this, doc| {
            Ok(tydecode::parse_substs_data(doc.data,
                                        dcx.cdata.cnum,
                                        doc.start,
                                        dcx.tcx,
                                        |s, a| this.convert_def_id(dcx, s, a)))
        }).unwrap()
    }

    fn read_auto_adjustment<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                                    -> ty::AutoAdjustment<'tcx> {
        self.read_enum("AutoAdjustment", |this| {
            let variants = ["AutoAddEnv", "AutoDerefRef"];
            this.read_enum_variant(&variants, |this, i| {
                Ok(match i {
                    1 => {
                        let def_id: ast::DefId =
                            this.read_def_id(dcx);

                        ty::AdjustReifyFnPointer(def_id)
                    }
                    2 => {
                        let auto_deref_ref: ty::AutoDerefRef =
                            this.read_enum_variant_arg(0,
                                |this| Ok(this.read_auto_deref_ref(dcx))).unwrap();

                        ty::AdjustDerefRef(auto_deref_ref)
                    }
                    _ => panic!("bad enum variant for ty::AutoAdjustment")
                })
            })
        }).unwrap()
    }

    fn read_auto_deref_ref<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                                   -> ty::AutoDerefRef<'tcx> {
        self.read_struct("AutoDerefRef", 2, |this| {
            Ok(ty::AutoDerefRef {
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
            })
        }).unwrap()
    }

    fn read_autoref<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>) -> ty::AutoRef<'tcx> {
        self.read_enum("AutoRef", |this| {
            let variants = ["AutoPtr",
                            "AutoUnsize",
                            "AutoUnsizeUniq",
                            "AutoUnsafe"];
            this.read_enum_variant(&variants, |this, i| {
                Ok(match i {
                    0 => {
                        let r: ty::Region =
                            this.read_enum_variant_arg(0, |this| Decodable::decode(this)).unwrap();
                        let m: ast::Mutability =
                            this.read_enum_variant_arg(1, |this| Decodable::decode(this)).unwrap();
                        let a: Option<Box<ty::AutoRef>> =
                            this.read_enum_variant_arg(2, |this| this.read_option(|this, b| {
                                if b {
                                    Ok(Some(box this.read_autoref(dcx)))
                                } else {
                                    Ok(None)
                                }
                            })).unwrap();

                        ty::AutoPtr(r.tr(dcx), m, a)
                    }
                    1 => {
                        let uk: ty::UnsizeKind =
                            this.read_enum_variant_arg(0,
                                |this| Ok(this.read_unsize_kind(dcx))).unwrap();

                        ty::AutoUnsize(uk)
                    }
                    2 => {
                        let uk: ty::UnsizeKind =
                            this.read_enum_variant_arg(0,
                                |this| Ok(this.read_unsize_kind(dcx))).unwrap();

                        ty::AutoUnsizeUniq(uk)
                    }
                    3 => {
                        let m: ast::Mutability =
                            this.read_enum_variant_arg(0, |this| Decodable::decode(this)).unwrap();
                        let a: Option<Box<ty::AutoRef>> =
                            this.read_enum_variant_arg(1, |this| this.read_option(|this, b| {
                                if b {
                                    Ok(Some(box this.read_autoref(dcx)))
                                } else {
                                    Ok(None)
                                }
                            })).unwrap();

                        ty::AutoUnsafe(m, a)
                    }
                    _ => panic!("bad enum variant for ty::AutoRef")
                })
            })
        }).unwrap()
    }

    fn read_unsize_kind<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                                -> ty::UnsizeKind<'tcx> {
        self.read_enum("UnsizeKind", |this| {
            let variants = &["UnsizeLength", "UnsizeStruct", "UnsizeVtable"];
            this.read_enum_variant(variants, |this, i| {
                Ok(match i {
                    0 => {
                        let len: uint =
                            this.read_enum_variant_arg(0, |this| Decodable::decode(this)).unwrap();

                        ty::UnsizeLength(len)
                    }
                    1 => {
                        let uk: ty::UnsizeKind =
                            this.read_enum_variant_arg(0,
                                |this| Ok(this.read_unsize_kind(dcx))).unwrap();
                        let idx: uint =
                            this.read_enum_variant_arg(1, |this| Decodable::decode(this)).unwrap();

                        ty::UnsizeStruct(box uk, idx)
                    }
                    2 => {
                        let ty_trait = try!(this.read_enum_variant_arg(0, |this| {
                            let principal = try!(this.read_struct_field("principal", 0, |this| {
                                Ok(this.read_poly_trait_ref(dcx))
                            }));
                            Ok(ty::TyTrait {
                                principal: principal,
                                bounds: try!(this.read_struct_field("bounds", 1, |this| {
                                    Ok(this.read_existential_bounds(dcx))
                                })),
                            })
                        }));
                        let self_ty =
                            this.read_enum_variant_arg(1, |this| Ok(this.read_ty(dcx))).unwrap();
                        ty::UnsizeVtable(ty_trait, self_ty)
                    }
                    _ => panic!("bad enum variant for ty::UnsizeKind")
                })
            })
        }).unwrap()
    }

    fn read_closure_kind<'b, 'c>(&mut self, _dcx: &DecodeContext<'b, 'c, 'tcx>)
                                 -> ty::ClosureKind
    {
        Decodable::decode(self).ok().unwrap()
    }

    fn read_closure_ty<'b, 'c>(&mut self, dcx: &DecodeContext<'b, 'c, 'tcx>)
                               -> ty::ClosureTy<'tcx>
    {
        self.read_opaque(|this, doc| {
            Ok(tydecode::parse_ty_closure_data(
                doc.data,
                dcx.cdata.cnum,
                doc.start,
                dcx.tcx,
                |s, a| this.convert_def_id(dcx, s, a)))
        }).unwrap()
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
    /// Unboxed closures are cloned along with the function being
    /// inlined, and all side tables use interned node IDs, so we
    /// translate their def IDs accordingly.
    ///
    /// It'd be really nice to refactor the type repr to not include
    /// def-ids so that all these distinctions were unnecessary.
    fn convert_def_id(&mut self,
                      dcx: &DecodeContext,
                      source: tydecode::DefIdSource,
                      did: ast::DefId)
                      -> ast::DefId {
        let r = match source {
            NominalType | TypeWithId | RegionParameter => dcx.tr_def_id(did),
            TypeParameter | ClosureSource => dcx.tr_intern_def_id(did)
        };
        debug!("convert_def_id(source={:?}, did={:?})={:?}", source, did, r);
        return r;
    }
}

fn decode_side_tables(dcx: &DecodeContext,
                      ast_doc: rbml::Doc) {
    let tbl_doc = ast_doc.get(c::tag_table as uint);
    reader::docs(tbl_doc, |tag, entry_doc| {
        let id0 = entry_doc.get(c::tag_table_id as uint).as_int();
        let id = dcx.tr_id(id0 as ast::NodeId);

        debug!(">> Side table document with tag 0x{:x} \
                found for id {} (orig {})",
               tag, id, id0);
        let decoded_tag: Option<c::astencode_tag> = FromPrimitive::from_uint(tag);
        match decoded_tag {
            None => {
                dcx.tcx.sess.bug(
                    &format!("unknown tag found in side tables: {:x}",
                            tag)[]);
            }
            Some(value) => {
                let val_doc = entry_doc.get(c::tag_table_val as uint);
                let mut val_dsr = reader::Decoder::new(val_doc);
                let val_dsr = &mut val_dsr;

                match value {
                    c::tag_table_def => {
                        let def = decode_def(dcx, val_doc);
                        dcx.tcx.def_map.borrow_mut().insert(id, def);
                    }
                    c::tag_table_node_type => {
                        let ty = val_dsr.read_ty(dcx);
                        debug!("inserting ty for node {}: {}",
                               id, ty_to_string(dcx.tcx, ty));
                        dcx.tcx.node_types.borrow_mut().insert(id, ty);
                    }
                    c::tag_table_item_subst => {
                        let item_substs = ty::ItemSubsts {
                            substs: val_dsr.read_substs(dcx)
                        };
                        dcx.tcx.item_substs.borrow_mut().insert(
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
                        let ub: ty::UpvarCapture = Decodable::decode(val_dsr).unwrap();
                        dcx.tcx.upvar_capture_map.borrow_mut().insert(upvar_id, ub.tr(dcx));
                    }
                    c::tag_table_tcache => {
                        let type_scheme = val_dsr.read_type_scheme(dcx);
                        let lid = ast::DefId { krate: ast::LOCAL_CRATE, node: id };
                        dcx.tcx.tcache.borrow_mut().insert(lid, type_scheme);
                    }
                    c::tag_table_param_defs => {
                        let bounds = val_dsr.read_type_param_def(dcx);
                        dcx.tcx.ty_param_defs.borrow_mut().insert(id, bounds);
                    }
                    c::tag_table_method_map => {
                        let (adjustment, method) = val_dsr.read_method_callee(dcx);
                        let method_call = MethodCall {
                            expr_id: id,
                            adjustment: adjustment
                        };
                        dcx.tcx.method_map.borrow_mut().insert(method_call, method);
                    }
                    c::tag_table_object_cast_map => {
                        let trait_ref = val_dsr.read_poly_trait_ref(dcx);
                        dcx.tcx.object_cast_map.borrow_mut()
                                               .insert(id, trait_ref);
                    }
                    c::tag_table_adjustments => {
                        let adj: ty::AutoAdjustment = val_dsr.read_auto_adjustment(dcx);
                        dcx.tcx.adjustments.borrow_mut().insert(id, adj);
                    }
                    c::tag_table_closure_tys => {
                        let closure_ty =
                            val_dsr.read_closure_ty(dcx);
                        dcx.tcx.closure_tys.borrow_mut().insert(ast_util::local_def(id),
                                                                closure_ty);
                    }
                    c::tag_table_closure_kinds => {
                        let closure_kind =
                            val_dsr.read_closure_kind(dcx);
                        dcx.tcx.closure_kinds.borrow_mut().insert(ast_util::local_def(id),
                                                                  closure_kind);
                    }
                    c::tag_table_const_qualif => {
                        let qualif: ConstQualif = Decodable::decode(val_dsr).unwrap();
                        dcx.tcx.const_qualif_map.borrow_mut().insert(id, qualif);
                    }
                    _ => {
                        dcx.tcx.sess.bug(
                            &format!("unknown tag found in side tables: {:x}",
                                    tag)[]);
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
fn encode_item_ast(rbml_w: &mut Encoder, item: &ast::Item) {
    rbml_w.start_tag(c::tag_tree as uint);
    (*item).encode(rbml_w);
    rbml_w.end_tag();
}

#[cfg(test)]
fn decode_item_ast(par_doc: rbml::Doc) -> ast::Item {
    let chi_doc = par_doc.get(c::tag_tree as uint);
    let mut d = reader::Decoder::new(chi_doc);
    Decodable::decode(&mut d).unwrap()
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
            expn_id: codemap::NO_EXPANSION
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
fn roundtrip(in_item: Option<P<ast::Item>>) {
    let in_item = in_item.unwrap();
    let mut wr = SeekableMemWriter::new();
    encode_item_ast(&mut writer::Encoder::new(&mut wr), &*in_item);
    let rbml_doc = rbml::Doc::new(wr.get_ref());
    let out_item = decode_item_ast(rbml_doc);

    assert!(*in_item == out_item);
}

#[test]
fn test_basic() {
    let cx = mk_ctxt();
    roundtrip(quote_item!(&cx,
        fn foo() {}
    ));
}
/* NOTE: When there's a snapshot, update this (yay quasiquoter!)
#[test]
fn test_smalltalk() {
    let cx = mk_ctxt();
    roundtrip(quote_item!(&cx,
        fn foo() -> int { 3 + 4 } // first smalltalk program ever executed.
    ));
}
*/

#[test]
fn test_more() {
    let cx = mk_ctxt();
    roundtrip(quote_item!(&cx,
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
    let item_exp = ast::IIItem(quote_item!(&cx,
        fn new_int_alist<B>() -> alist<int, B> {
            return alist {eq_fn: eq_int, data: Vec::new()};
        }
    ).unwrap());
    match (item_out, item_exp) {
      (ast::IIItem(item_out), ast::IIItem(item_exp)) => {
        assert!(pprust::item_to_string(&*item_out) ==
                pprust::item_to_string(&*item_exp));
      }
      _ => panic!()
    }
}
