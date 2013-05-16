// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use c = metadata::common;
use cstore = metadata::cstore;
use driver::session::Session;
use e = metadata::encoder;
use metadata::decoder;
use metadata::encoder;
use metadata::tydecode;
use metadata::tydecode::{DefIdSource, NominalType, TypeWithId, TypeParameter};
use metadata::tyencode;
use middle::freevars::freevar_entry;
use middle::typeck::{method_origin, method_map_entry};
use middle::{ty, typeck, moves};
use middle;
use util::ppaux::ty_to_str;

use std::ebml::reader;
use std::ebml;
use std::serialize;
use std::serialize::{Encoder, Encodable, EncoderHelpers, DecoderHelpers};
use std::serialize::{Decoder, Decodable};
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::inlined_item_utils;
use syntax::ast_util;
use syntax::codemap::span;
use syntax::codemap;
use syntax::fold::*;
use syntax::fold;
use syntax;
use writer = std::ebml::writer;

#[cfg(test)] use syntax::parse;
#[cfg(test)] use syntax::print::pprust;

// Auxiliary maps of things to be encoded
pub struct Maps {
    root_map: middle::borrowck::root_map,
    method_map: middle::typeck::method_map,
    vtable_map: middle::typeck::vtable_map,
    write_guard_map: middle::borrowck::write_guard_map,
    moves_map: middle::moves::MovesMap,
    capture_map: middle::moves::CaptureMap,
}

struct DecodeContext {
    cdata: @cstore::crate_metadata,
    tcx: ty::ctxt,
    maps: Maps
}

struct ExtendedDecodeContext {
    dcx: @DecodeContext,
    from_id_range: ast_util::id_range,
    to_id_range: ast_util::id_range
}

trait tr {
    fn tr(&self, xcx: @ExtendedDecodeContext) -> Self;
}

trait tr_intern {
    fn tr_intern(&self, xcx: @ExtendedDecodeContext) -> ast::def_id;
}

// ______________________________________________________________________
// Top-level methods.

pub fn encode_inlined_item(ecx: @e::EncodeContext,
                           ebml_w: &mut writer::Encoder,
                           path: &[ast_map::path_elt],
                           ii: ast::inlined_item,
                           maps: Maps) {
    debug!("> Encoding inlined item: %s::%s (%u)",
           ast_map::path_to_str(path, ecx.tcx.sess.parse_sess.interner),
           *ecx.tcx.sess.str_of(ii.ident()),
           ebml_w.writer.tell());

    let id_range = ast_util::compute_id_range_for_inlined_item(&ii);

    ebml_w.start_tag(c::tag_ast as uint);
    id_range.encode(ebml_w);
    encode_ast(ebml_w, simplify_ast(&ii));
    encode_side_tables_for_ii(ecx, maps, ebml_w, &ii);
    ebml_w.end_tag();

    debug!("< Encoded inlined fn: %s::%s (%u)",
           ast_map::path_to_str(path, ecx.tcx.sess.parse_sess.interner),
           *ecx.tcx.sess.str_of(ii.ident()),
           ebml_w.writer.tell());
}

pub fn decode_inlined_item(cdata: @cstore::crate_metadata,
                           tcx: ty::ctxt,
                           maps: Maps,
                           path: ast_map::path,
                           par_doc: ebml::Doc)
                        -> Option<ast::inlined_item> {
    let dcx = @DecodeContext {
        cdata: cdata,
        tcx: tcx,
        maps: maps
    };
    match par_doc.opt_child(c::tag_ast) {
      None => None,
      Some(ast_doc) => {
        debug!("> Decoding inlined fn: %s::?",
               ast_map::path_to_str(path, tcx.sess.parse_sess.interner));
        let mut ast_dsr = reader::Decoder(ast_doc);
        let from_id_range = Decodable::decode(&mut ast_dsr);
        let to_id_range = reserve_id_range(dcx.tcx.sess, from_id_range);
        let xcx = @ExtendedDecodeContext {
            dcx: dcx,
            from_id_range: from_id_range,
            to_id_range: to_id_range
        };
        let raw_ii = decode_ast(ast_doc);
        let ii = renumber_ast(xcx, raw_ii);
        debug!("Fn named: %s", *tcx.sess.str_of(ii.ident()));
        debug!("< Decoded inlined fn: %s::%s",
               ast_map::path_to_str(path, tcx.sess.parse_sess.interner),
               *tcx.sess.str_of(ii.ident()));
        ast_map::map_decoded_item(tcx.sess.diagnostic(),
                                  dcx.tcx.items, path, &ii);
        decode_side_tables(xcx, ast_doc);
        match ii {
          ast::ii_item(i) => {
            debug!(">>> DECODED ITEM >>>\n%s\n<<< DECODED ITEM <<<",
                   syntax::print::pprust::item_to_str(i, tcx.sess.intr()));
          }
          _ => { }
        }
        Some(ii)
      }
    }
}

// ______________________________________________________________________
// Enumerating the IDs which appear in an AST

fn reserve_id_range(sess: Session,
                    from_id_range: ast_util::id_range) -> ast_util::id_range {
    // Handle the case of an empty range:
    if from_id_range.empty() { return from_id_range; }
    let cnt = from_id_range.max - from_id_range.min;
    let to_id_min = sess.parse_sess.next_id;
    let to_id_max = sess.parse_sess.next_id + cnt;
    sess.parse_sess.next_id = to_id_max;
    ast_util::id_range { min: to_id_min, max: to_id_min }
}

pub impl ExtendedDecodeContext {
    fn tr_id(&self, id: ast::node_id) -> ast::node_id {
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
    fn tr_def_id(&self, did: ast::def_id) -> ast::def_id {
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
    fn tr_intern_def_id(&self, did: ast::def_id) -> ast::def_id {
        /*!
         * Translates an INTERNAL def-id, meaning a def-id that is
         * known to refer to some part of the item currently being
         * inlined.  In that case, we want to convert the def-id to
         * refer to the current crate and to the new, inlined node-id.
         */

        assert!(did.crate == ast::local_crate);
        ast::def_id { crate: ast::local_crate, node: self.tr_id(did.node) }
    }
    fn tr_span(&self, _span: span) -> span {
        codemap::dummy_sp() // FIXME (#1972): handle span properly
    }
}

impl tr_intern for ast::def_id {
    fn tr_intern(&self, xcx: @ExtendedDecodeContext) -> ast::def_id {
        xcx.tr_intern_def_id(*self)
    }
}

impl tr for ast::def_id {
    fn tr(&self, xcx: @ExtendedDecodeContext) -> ast::def_id {
        xcx.tr_def_id(*self)
    }
}

impl tr for span {
    fn tr(&self, xcx: @ExtendedDecodeContext) -> span {
        xcx.tr_span(*self)
    }
}

trait def_id_encoder_helpers {
    fn emit_def_id(&mut self, did: ast::def_id);
}

impl<S:serialize::Encoder> def_id_encoder_helpers for S {
    fn emit_def_id(&mut self, did: ast::def_id) {
        did.encode(self)
    }
}

trait def_id_decoder_helpers {
    fn read_def_id(&mut self, xcx: @ExtendedDecodeContext) -> ast::def_id;
}

impl<D:serialize::Decoder> def_id_decoder_helpers for D {
    fn read_def_id(&mut self, xcx: @ExtendedDecodeContext) -> ast::def_id {
        let did: ast::def_id = Decodable::decode(self);
        did.tr(xcx)
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

fn encode_ast(ebml_w: &mut writer::Encoder, item: ast::inlined_item) {
    ebml_w.start_tag(c::tag_tree as uint);
    item.encode(ebml_w);
    ebml_w.end_tag();
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
fn simplify_ast(ii: &ast::inlined_item) -> ast::inlined_item {
    fn drop_nested_items(blk: &ast::blk_, fld: @fold::ast_fold) -> ast::blk_ {
        let stmts_sans_items = do blk.stmts.filtered |stmt| {
            match stmt.node {
              ast::stmt_expr(_, _) | ast::stmt_semi(_, _) |
              ast::stmt_decl(@codemap::spanned { node: ast::decl_local(_),
                                             span: _}, _) => true,
              ast::stmt_decl(@codemap::spanned { node: ast::decl_item(_),
                                             span: _}, _) => false,
              ast::stmt_mac(*) => fail!("unexpanded macro in astencode")
            }
        };
        let blk_sans_items = ast::blk_ {
            view_items: ~[], // I don't know if we need the view_items here,
                             // but it doesn't break tests!
            stmts: stmts_sans_items,
            expr: blk.expr,
            id: blk.id,
            rules: blk.rules
        };
        fold::noop_fold_block(&blk_sans_items, fld)
    }

    let fld = fold::make_fold(@fold::AstFoldFns {
        fold_block: fold::wrap(drop_nested_items),
        .. *fold::default_ast_fold()
    });

    match *ii {
      ast::ii_item(i) => {
        ast::ii_item(fld.fold_item(i).get()) //hack: we're not dropping items
      }
      ast::ii_method(d, m) => {
        ast::ii_method(d, fld.fold_method(m))
      }
      ast::ii_foreign(i) => {
        ast::ii_foreign(fld.fold_foreign_item(i))
      }
    }
}

fn decode_ast(par_doc: ebml::Doc) -> ast::inlined_item {
    let chi_doc = par_doc.get(c::tag_tree as uint);
    let mut d = reader::Decoder(chi_doc);
    Decodable::decode(&mut d)
}

fn renumber_ast(xcx: @ExtendedDecodeContext, ii: ast::inlined_item)
    -> ast::inlined_item {
    let fld = fold::make_fold(@fold::AstFoldFns{
        new_id: |a| xcx.tr_id(a),
        new_span: |a| xcx.tr_span(a),
        .. *fold::default_ast_fold()
    });

    match ii {
      ast::ii_item(i) => {
        ast::ii_item(fld.fold_item(i).get())
      }
      ast::ii_method(d, m) => {
        ast::ii_method(xcx.tr_def_id(d), fld.fold_method(m))
      }
      ast::ii_foreign(i) => {
        ast::ii_foreign(fld.fold_foreign_item(i))
      }
     }
}

// ______________________________________________________________________
// Encoding and decoding of ast::def

fn encode_def(ebml_w: &mut writer::Encoder, def: ast::def) {
    def.encode(ebml_w)
}

fn decode_def(xcx: @ExtendedDecodeContext, doc: ebml::Doc) -> ast::def {
    let mut dsr = reader::Decoder(doc);
    let def: ast::def = Decodable::decode(&mut dsr);
    def.tr(xcx)
}

impl tr for ast::def {
    fn tr(&self, xcx: @ExtendedDecodeContext) -> ast::def {
        match *self {
          ast::def_fn(did, p) => { ast::def_fn(did.tr(xcx), p) }
          ast::def_static_method(did, did2_opt, p) => {
            ast::def_static_method(did.tr(xcx),
                                   did2_opt.map(|did2| did2.tr(xcx)),
                                   p)
          }
          ast::def_self_ty(nid) => { ast::def_self_ty(xcx.tr_id(nid)) }
          ast::def_self(nid, i) => { ast::def_self(xcx.tr_id(nid), i) }
          ast::def_mod(did) => { ast::def_mod(did.tr(xcx)) }
          ast::def_foreign_mod(did) => { ast::def_foreign_mod(did.tr(xcx)) }
          ast::def_const(did) => { ast::def_const(did.tr(xcx)) }
          ast::def_arg(nid, b) => { ast::def_arg(xcx.tr_id(nid), b) }
          ast::def_local(nid, b) => { ast::def_local(xcx.tr_id(nid), b) }
          ast::def_variant(e_did, v_did) => {
            ast::def_variant(e_did.tr(xcx), v_did.tr(xcx))
          }
          ast::def_trait(did) => ast::def_trait(did.tr(xcx)),
          ast::def_ty(did) => ast::def_ty(did.tr(xcx)),
          ast::def_prim_ty(p) => ast::def_prim_ty(p),
          ast::def_ty_param(did, v) => ast::def_ty_param(did.tr(xcx), v),
          ast::def_binding(nid, bm) => ast::def_binding(xcx.tr_id(nid), bm),
          ast::def_use(did) => ast::def_use(did.tr(xcx)),
          ast::def_upvar(nid1, def, nid2, nid3) => {
            ast::def_upvar(xcx.tr_id(nid1),
                           @(*def).tr(xcx),
                           xcx.tr_id(nid2),
                           xcx.tr_id(nid3))
          }
          ast::def_struct(did) => {
            ast::def_struct(did.tr(xcx))
          }
          ast::def_region(nid) => ast::def_region(xcx.tr_id(nid)),
          ast::def_typaram_binder(nid) => {
            ast::def_typaram_binder(xcx.tr_id(nid))
          }
          ast::def_label(nid) => ast::def_label(xcx.tr_id(nid))
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of adjustment information

impl tr for ty::AutoAdjustment {
    fn tr(&self, xcx: @ExtendedDecodeContext) -> ty::AutoAdjustment {
        match self {
            &ty::AutoAddEnv(r, s) => {
                ty::AutoAddEnv(r.tr(xcx), s)
            }

            &ty::AutoDerefRef(ref adr) => {
                ty::AutoDerefRef(ty::AutoDerefRef {
                    autoderefs: adr.autoderefs,
                    autoref: adr.autoref.map(|ar| ar.tr(xcx)),
                })
            }
        }
    }
}

impl tr for ty::AutoRef {
    fn tr(&self, xcx: @ExtendedDecodeContext) -> ty::AutoRef {
        self.map_region(|r| r.tr(xcx))
    }
}

impl tr for ty::Region {
    fn tr(&self, xcx: @ExtendedDecodeContext) -> ty::Region {
        match *self {
            ty::re_bound(br) => ty::re_bound(br.tr(xcx)),
            ty::re_scope(id) => ty::re_scope(xcx.tr_id(id)),
            ty::re_empty | ty::re_static | ty::re_infer(*) => *self,
            ty::re_free(ref fr) => {
                ty::re_free(ty::FreeRegion {scope_id: xcx.tr_id(fr.scope_id),
                                            bound_region: fr.bound_region.tr(xcx)})
            }
        }
    }
}

impl tr for ty::bound_region {
    fn tr(&self, xcx: @ExtendedDecodeContext) -> ty::bound_region {
        match *self {
            ty::br_anon(_) | ty::br_named(_) | ty::br_self |
            ty::br_fresh(_) => *self,
            ty::br_cap_avoid(id, br) => ty::br_cap_avoid(xcx.tr_id(id),
                                                         @br.tr(xcx))
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of freevar information

fn encode_freevar_entry(ebml_w: &mut writer::Encoder, fv: @freevar_entry) {
    (*fv).encode(ebml_w)
}

trait ebml_decoder_helper {
    fn read_freevar_entry(&mut self, xcx: @ExtendedDecodeContext)
                          -> freevar_entry;
}

impl ebml_decoder_helper for reader::Decoder {
    fn read_freevar_entry(&mut self, xcx: @ExtendedDecodeContext)
                          -> freevar_entry {
        let fv: freevar_entry = Decodable::decode(self);
        fv.tr(xcx)
    }
}

impl tr for freevar_entry {
    fn tr(&self, xcx: @ExtendedDecodeContext) -> freevar_entry {
        freevar_entry {
            def: self.def.tr(xcx),
            span: self.span.tr(xcx),
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of CaptureVar information

trait capture_var_helper {
    fn read_capture_var(&mut self, xcx: @ExtendedDecodeContext)
                        -> moves::CaptureVar;
}

impl capture_var_helper for reader::Decoder {
    fn read_capture_var(&mut self, xcx: @ExtendedDecodeContext)
                        -> moves::CaptureVar {
        let cvar: moves::CaptureVar = Decodable::decode(self);
        cvar.tr(xcx)
    }
}

impl tr for moves::CaptureVar {
    fn tr(&self, xcx: @ExtendedDecodeContext) -> moves::CaptureVar {
        moves::CaptureVar {
            def: self.def.tr(xcx),
            span: self.span.tr(xcx),
            mode: self.mode
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of method_map_entry

trait read_method_map_entry_helper {
    fn read_method_map_entry(&mut self, xcx: @ExtendedDecodeContext)
                             -> method_map_entry;
}

fn encode_method_map_entry(ecx: @e::EncodeContext,
                           ebml_w: &mut writer::Encoder,
                           mme: method_map_entry) {
    do ebml_w.emit_struct("method_map_entry", 3) |ebml_w| {
        do ebml_w.emit_struct_field("self_ty", 0u) |ebml_w| {
            ebml_w.emit_ty(ecx, mme.self_ty);
        }
        do ebml_w.emit_struct_field("explicit_self", 2u) |ebml_w| {
            mme.explicit_self.encode(ebml_w);
        }
        do ebml_w.emit_struct_field("origin", 1u) |ebml_w| {
            mme.origin.encode(ebml_w);
        }
        do ebml_w.emit_struct_field("self_mode", 3) |ebml_w| {
            mme.self_mode.encode(ebml_w);
        }
    }
}

impl read_method_map_entry_helper for reader::Decoder {
    fn read_method_map_entry(&mut self, xcx: @ExtendedDecodeContext)
                             -> method_map_entry {
        do self.read_struct("method_map_entry", 3) |this| {
            method_map_entry {
                self_ty: this.read_struct_field("self_ty", 0u, |this| {
                    this.read_ty(xcx)
                }),
                explicit_self: this.read_struct_field("explicit_self",
                                                      2,
                                                      |this| {
                    let explicit_self: ast::explicit_self_ = Decodable::decode(this);
                    explicit_self
                }),
                origin: this.read_struct_field("origin", 1, |this| {
                    let method_origin: method_origin =
                        Decodable::decode(this);
                    method_origin.tr(xcx)
                }),
                self_mode: this.read_struct_field("self_mode", 3, |this| {
                    let self_mode: ty::SelfMode = Decodable::decode(this);
                    self_mode
                }),
            }
        }
    }
}

impl tr for method_origin {
    fn tr(&self, xcx: @ExtendedDecodeContext) -> method_origin {
        match *self {
          typeck::method_static(did) => {
              typeck::method_static(did.tr(xcx))
          }
          typeck::method_param(ref mp) => {
            typeck::method_param(
                typeck::method_param {
                    trait_id: mp.trait_id.tr(xcx),
                    .. *mp
                }
            )
          }
          typeck::method_trait(did, m, vstore) => {
              typeck::method_trait(did.tr(xcx), m, vstore)
          }
          typeck::method_self(did, m) => {
              typeck::method_self(did.tr(xcx), m)
          }
          typeck::method_super(trait_did, m) => {
              typeck::method_super(trait_did.tr(xcx), m)
          }
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding vtable_res

fn encode_vtable_res(ecx: @e::EncodeContext,
                     ebml_w: &mut writer::Encoder,
                     dr: typeck::vtable_res) {
    // can't autogenerate this code because automatic code of
    // ty::t doesn't work, and there is no way (atm) to have
    // hand-written encoding routines combine with auto-generated
    // ones.  perhaps we should fix this.
    do ebml_w.emit_from_vec(*dr) |ebml_w, vtable_origin| {
        encode_vtable_origin(ecx, ebml_w, vtable_origin)
    }
}

fn encode_vtable_origin(ecx: @e::EncodeContext,
                        ebml_w: &mut writer::Encoder,
                        vtable_origin: &typeck::vtable_origin) {
    do ebml_w.emit_enum(~"vtable_origin") |ebml_w| {
        match *vtable_origin {
          typeck::vtable_static(def_id, ref tys, vtable_res) => {
            do ebml_w.emit_enum_variant(~"vtable_static", 0u, 3u) |ebml_w| {
                do ebml_w.emit_enum_variant_arg(0u) |ebml_w| {
                    ebml_w.emit_def_id(def_id)
                }
                do ebml_w.emit_enum_variant_arg(1u) |ebml_w| {
                    ebml_w.emit_tys(ecx, /*bad*/copy *tys);
                }
                do ebml_w.emit_enum_variant_arg(2u) |ebml_w| {
                    encode_vtable_res(ecx, ebml_w, vtable_res);
                }
            }
          }
          typeck::vtable_param(pn, bn) => {
            do ebml_w.emit_enum_variant(~"vtable_param", 1u, 2u) |ebml_w| {
                do ebml_w.emit_enum_variant_arg(0u) |ebml_w| {
                    ebml_w.emit_uint(pn);
                }
                do ebml_w.emit_enum_variant_arg(1u) |ebml_w| {
                    ebml_w.emit_uint(bn);
                }
            }
          }
        }
    }
}

trait vtable_decoder_helpers {
    fn read_vtable_res(&mut self, xcx: @ExtendedDecodeContext)
                      -> typeck::vtable_res;
    fn read_vtable_origin(&mut self, xcx: @ExtendedDecodeContext)
                          -> typeck::vtable_origin;
}

impl vtable_decoder_helpers for reader::Decoder {
    fn read_vtable_res(&mut self, xcx: @ExtendedDecodeContext)
                      -> typeck::vtable_res {
        @self.read_to_vec(|this| this.read_vtable_origin(xcx))
    }

    fn read_vtable_origin(&mut self, xcx: @ExtendedDecodeContext)
        -> typeck::vtable_origin {
        do self.read_enum("vtable_origin") |this| {
            do this.read_enum_variant(["vtable_static", "vtable_param"])
                    |this, i| {
                match i {
                  0 => {
                    typeck::vtable_static(
                        do this.read_enum_variant_arg(0u) |this| {
                            this.read_def_id(xcx)
                        },
                        do this.read_enum_variant_arg(1u) |this| {
                            this.read_tys(xcx)
                        },
                        do this.read_enum_variant_arg(2u) |this| {
                            this.read_vtable_res(xcx)
                        }
                    )
                  }
                  1 => {
                    typeck::vtable_param(
                        do this.read_enum_variant_arg(0u) |this| {
                            this.read_uint()
                        },
                        do this.read_enum_variant_arg(1u) |this| {
                            this.read_uint()
                        }
                    )
                  }
                  // hard to avoid - user input
                  _ => fail!("bad enum variant")
                }
            }
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding the side tables

trait get_ty_str_ctxt {
    fn ty_str_ctxt(@self) -> @tyencode::ctxt;
}

impl get_ty_str_ctxt for e::EncodeContext {
    // IMPLICIT SELF WARNING: fix this!
    fn ty_str_ctxt(@self) -> @tyencode::ctxt {
        @tyencode::ctxt {diag: self.tcx.sess.diagnostic(),
                        ds: e::def_to_str,
                        tcx: self.tcx,
                        reachable: |a| encoder::reachable(self, a),
                        abbrevs: tyencode::ac_use_abbrevs(self.type_abbrevs)}
    }
}

trait ebml_writer_helpers {
    fn emit_ty(&mut self, ecx: @e::EncodeContext, ty: ty::t);
    fn emit_vstore(&mut self, ecx: @e::EncodeContext, vstore: ty::vstore);
    fn emit_tys(&mut self, ecx: @e::EncodeContext, tys: &[ty::t]);
    fn emit_type_param_def(&mut self,
                           ecx: @e::EncodeContext,
                           type_param_def: &ty::TypeParameterDef);
    fn emit_tpbt(&mut self,
                 ecx: @e::EncodeContext,
                 tpbt: ty::ty_param_bounds_and_ty);
}

impl ebml_writer_helpers for writer::Encoder {
    fn emit_ty(&mut self, ecx: @e::EncodeContext, ty: ty::t) {
        do self.emit_opaque |this| {
            e::write_type(ecx, this, ty)
        }
    }

    fn emit_vstore(&mut self, ecx: @e::EncodeContext, vstore: ty::vstore) {
        do self.emit_opaque |this| {
            e::write_vstore(ecx, this, vstore)
        }
    }

    fn emit_tys(&mut self, ecx: @e::EncodeContext, tys: &[ty::t]) {
        do self.emit_from_vec(tys) |this, ty| {
            this.emit_ty(ecx, *ty)
        }
    }

    fn emit_type_param_def(&mut self,
                           ecx: @e::EncodeContext,
                           type_param_def: &ty::TypeParameterDef) {
        do self.emit_opaque |this| {
            tyencode::enc_type_param_def(this.writer,
                                         ecx.ty_str_ctxt(),
                                         type_param_def)
        }
    }

    fn emit_tpbt(&mut self,
                 ecx: @e::EncodeContext,
                 tpbt: ty::ty_param_bounds_and_ty) {
        do self.emit_struct("ty_param_bounds_and_ty", 2) |this| {
            do this.emit_struct_field(~"generics", 0) |this| {
                do this.emit_struct("Generics", 2) |this| {
                    do this.emit_struct_field(~"type_param_defs", 0) |this| {
                        do this.emit_from_vec(*tpbt.generics.type_param_defs)
                                |this, type_param_def| {
                            this.emit_type_param_def(ecx, type_param_def);
                        }
                    }
                    do this.emit_struct_field(~"region_param", 1) |this| {
                        tpbt.generics.region_param.encode(this);
                    }
                }
            }
            do this.emit_struct_field(~"ty", 1) |this| {
                this.emit_ty(ecx, tpbt.ty);
            }
        }
    }
}

trait write_tag_and_id {
    fn tag(&mut self, tag_id: c::astencode_tag, f: &fn(&mut Self));
    fn id(&mut self, id: ast::node_id);
}

impl write_tag_and_id for writer::Encoder {
    fn tag(&mut self,
           tag_id: c::astencode_tag,
           f: &fn(&mut writer::Encoder)) {
        self.start_tag(tag_id as uint);
        f(self);
        self.end_tag();
    }

    fn id(&mut self, id: ast::node_id) {
        self.wr_tagged_u64(c::tag_table_id as uint, id as u64)
    }
}

fn encode_side_tables_for_ii(ecx: @e::EncodeContext,
                             maps: Maps,
                             ebml_w: &mut writer::Encoder,
                             ii: &ast::inlined_item) {
    ebml_w.start_tag(c::tag_table as uint);
    let new_ebml_w = copy *ebml_w;
    ast_util::visit_ids_for_inlined_item(
        ii,
        |id: ast::node_id| {
            // Note: this will cause a copy of ebml_w, which is bad as
            // it is mutable. But I believe it's harmless since we generate
            // balanced EBML.
            let mut new_ebml_w = copy new_ebml_w;
            encode_side_tables_for_id(ecx, maps, &mut new_ebml_w, id)
        });
    ebml_w.end_tag();
}

fn encode_side_tables_for_id(ecx: @e::EncodeContext,
                             maps: Maps,
                             ebml_w: &mut writer::Encoder,
                             id: ast::node_id) {
    let tcx = ecx.tcx;

    debug!("Encoding side tables for id %d", id);

    for tcx.def_map.find(&id).each |def| {
        do ebml_w.tag(c::tag_table_def) |ebml_w| {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) |ebml_w| {
                (*def).encode(ebml_w)
            }
        }
    }

    for tcx.node_types.find(&(id as uint)).each |&ty| {
        do ebml_w.tag(c::tag_table_node_type) |ebml_w| {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) |ebml_w| {
                ebml_w.emit_ty(ecx, *ty);
            }
        }
    }

    for tcx.node_type_substs.find(&id).each |tys| {
        do ebml_w.tag(c::tag_table_node_type_subst) |ebml_w| {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) |ebml_w| {
                ebml_w.emit_tys(ecx, **tys)
            }
        }
    }

    for tcx.freevars.find(&id).each |&fv| {
        do ebml_w.tag(c::tag_table_freevars) |ebml_w| {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) |ebml_w| {
                do ebml_w.emit_from_vec(**fv) |ebml_w, fv_entry| {
                    encode_freevar_entry(ebml_w, *fv_entry)
                }
            }
        }
    }

    let lid = ast::def_id { crate: ast::local_crate, node: id };
    for tcx.tcache.find(&lid).each |&tpbt| {
        do ebml_w.tag(c::tag_table_tcache) |ebml_w| {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) |ebml_w| {
                ebml_w.emit_tpbt(ecx, *tpbt);
            }
        }
    }

    for tcx.ty_param_defs.find(&id).each |&type_param_def| {
        do ebml_w.tag(c::tag_table_param_defs) |ebml_w| {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) |ebml_w| {
                ebml_w.emit_type_param_def(ecx, type_param_def)
            }
        }
    }

    for maps.method_map.find(&id).each |&mme| {
        do ebml_w.tag(c::tag_table_method_map) |ebml_w| {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) |ebml_w| {
                encode_method_map_entry(ecx, ebml_w, *mme)
            }
        }
    }

    for maps.vtable_map.find(&id).each |&dr| {
        do ebml_w.tag(c::tag_table_vtable_map) |ebml_w| {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) |ebml_w| {
                encode_vtable_res(ecx, ebml_w, *dr);
            }
        }
    }

    for tcx.adjustments.find(&id).each |adj| {
        do ebml_w.tag(c::tag_table_adjustments) |ebml_w| {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) |ebml_w| {
                (**adj).encode(ebml_w)
            }
        }
    }

    if maps.moves_map.contains(&id) {
        do ebml_w.tag(c::tag_table_moves_map) |ebml_w| {
            ebml_w.id(id);
        }
    }

    for maps.capture_map.find(&id).each |&cap_vars| {
        do ebml_w.tag(c::tag_table_capture_map) |ebml_w| {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) |ebml_w| {
                do ebml_w.emit_from_vec(*cap_vars) |ebml_w, cap_var| {
                    cap_var.encode(ebml_w);
                }
            }
        }
    }
}

trait doc_decoder_helpers {
    fn as_int(&self) -> int;
    fn opt_child(&self, tag: c::astencode_tag) -> Option<ebml::Doc>;
}

impl doc_decoder_helpers for ebml::Doc {
    fn as_int(&self) -> int { reader::doc_as_u64(*self) as int }
    fn opt_child(&self, tag: c::astencode_tag) -> Option<ebml::Doc> {
        reader::maybe_get_doc(*self, tag as uint)
    }
}

trait ebml_decoder_decoder_helpers {
    fn read_ty(&mut self, xcx: @ExtendedDecodeContext) -> ty::t;
    fn read_tys(&mut self, xcx: @ExtendedDecodeContext) -> ~[ty::t];
    fn read_type_param_def(&mut self, xcx: @ExtendedDecodeContext)
                           -> ty::TypeParameterDef;
    fn read_ty_param_bounds_and_ty(&mut self, xcx: @ExtendedDecodeContext)
                                -> ty::ty_param_bounds_and_ty;
    fn convert_def_id(&mut self,
                      xcx: @ExtendedDecodeContext,
                      source: DefIdSource,
                      did: ast::def_id)
                      -> ast::def_id;
}

impl ebml_decoder_decoder_helpers for reader::Decoder {
    fn read_ty(&mut self, xcx: @ExtendedDecodeContext) -> ty::t {
        // Note: regions types embed local node ids.  In principle, we
        // should translate these node ids into the new decode
        // context.  However, we do not bother, because region types
        // are not used during trans.

        return do self.read_opaque |this, doc| {
            let ty = tydecode::parse_ty_data(
                doc.data,
                xcx.dcx.cdata.cnum,
                doc.start,
                xcx.dcx.tcx,
                |s, a| this.convert_def_id(xcx, s, a));

            debug!("read_ty(%s) = %s",
                   type_string(doc),
                   ty_to_str(xcx.dcx.tcx, ty));

            ty
        };

        fn type_string(doc: ebml::Doc) -> ~str {
            let mut str = ~"";
            for uint::range(doc.start, doc.end) |i| {
                str::push_char(&mut str, doc.data[i] as char);
            }
            str
        }
    }

    fn read_tys(&mut self, xcx: @ExtendedDecodeContext) -> ~[ty::t] {
        self.read_to_vec(|this| this.read_ty(xcx) )
    }

    fn read_type_param_def(&mut self, xcx: @ExtendedDecodeContext)
                           -> ty::TypeParameterDef {
        do self.read_opaque |this, doc| {
            tydecode::parse_type_param_def_data(
                doc.data,
                doc.start,
                xcx.dcx.cdata.cnum,
                xcx.dcx.tcx,
                |s, a| this.convert_def_id(xcx, s, a))
        }
    }

    fn read_ty_param_bounds_and_ty(&mut self, xcx: @ExtendedDecodeContext)
                                   -> ty::ty_param_bounds_and_ty {
        do self.read_struct("ty_param_bounds_and_ty", 2) |this| {
            ty::ty_param_bounds_and_ty {
                generics: do this.read_struct_field("generics", 0) |this| {
                    do this.read_struct("Generics", 2) |this| {
                        ty::Generics {
                            type_param_defs:
                                this.read_struct_field("type_param_defs",
                                                       0,
                                                       |this| {
                                    @this.read_to_vec(|this|
                                        this.read_type_param_def(xcx))
                            }),
                            region_param:
                                this.read_struct_field("region_param",
                                                       1,
                                                       |this| {
                                    Decodable::decode(this)
                                })
                        }
                    }
                },
                ty: this.read_struct_field("ty", 1, |this| {
                    this.read_ty(xcx)
                })
            }
        }
    }

    fn convert_def_id(&mut self,
                      xcx: @ExtendedDecodeContext,
                      source: tydecode::DefIdSource,
                      did: ast::def_id)
                      -> ast::def_id {
        /*!
         *
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
         */

        let r = match source {
            NominalType | TypeWithId => xcx.tr_def_id(did),
            TypeParameter => xcx.tr_intern_def_id(did)
        };
        debug!("convert_def_id(source=%?, did=%?)=%?", source, did, r);
        return r;
    }
}

fn decode_side_tables(xcx: @ExtendedDecodeContext,
                      ast_doc: ebml::Doc) {
    let dcx = xcx.dcx;
    let tbl_doc = ast_doc.get(c::tag_table as uint);
    for reader::docs(tbl_doc) |tag, entry_doc| {
        let id0 = entry_doc.get(c::tag_table_id as uint).as_int();
        let id = xcx.tr_id(id0);

        debug!(">> Side table document with tag 0x%x \
                found for id %d (orig %d)",
               tag, id, id0);

        if tag == (c::tag_table_moves_map as uint) {
            dcx.maps.moves_map.insert(id);
        } else {
            let val_doc = entry_doc.get(c::tag_table_val as uint);
            let mut val_dsr = reader::Decoder(val_doc);
            let val_dsr = &mut val_dsr;
            if tag == (c::tag_table_def as uint) {
                let def = decode_def(xcx, val_doc);
                dcx.tcx.def_map.insert(id, def);
            } else if tag == (c::tag_table_node_type as uint) {
                let ty = val_dsr.read_ty(xcx);
                debug!("inserting ty for node %?: %s",
                       id, ty_to_str(dcx.tcx, ty));
                dcx.tcx.node_types.insert(id as uint, ty);
            } else if tag == (c::tag_table_node_type_subst as uint) {
                let tys = val_dsr.read_tys(xcx);
                dcx.tcx.node_type_substs.insert(id, tys);
            } else if tag == (c::tag_table_freevars as uint) {
                let fv_info = @val_dsr.read_to_vec(|val_dsr| {
                    @val_dsr.read_freevar_entry(xcx)
                });
                dcx.tcx.freevars.insert(id, fv_info);
            } else if tag == (c::tag_table_tcache as uint) {
                let tpbt = val_dsr.read_ty_param_bounds_and_ty(xcx);
                let lid = ast::def_id { crate: ast::local_crate, node: id };
                dcx.tcx.tcache.insert(lid, tpbt);
            } else if tag == (c::tag_table_param_defs as uint) {
                let bounds = val_dsr.read_type_param_def(xcx);
                dcx.tcx.ty_param_defs.insert(id, bounds);
            } else if tag == (c::tag_table_method_map as uint) {
                dcx.maps.method_map.insert(
                    id,
                    val_dsr.read_method_map_entry(xcx));
            } else if tag == (c::tag_table_vtable_map as uint) {
                dcx.maps.vtable_map.insert(id,
                                           val_dsr.read_vtable_res(xcx));
            } else if tag == (c::tag_table_adjustments as uint) {
                let adj: @ty::AutoAdjustment = @Decodable::decode(val_dsr);
                adj.tr(xcx);
                dcx.tcx.adjustments.insert(id, adj);
            } else if tag == (c::tag_table_capture_map as uint) {
                let cvars =
                    at_vec::to_managed_consume(
                        val_dsr.read_to_vec(
                            |val_dsr| val_dsr.read_capture_var(xcx)));
                dcx.maps.capture_map.insert(id, cvars);
            } else {
                xcx.dcx.tcx.sess.bug(
                    fmt!("unknown tag found in side tables: %x", tag));
            }
        }

        debug!(">< Side table doc loaded");
    }
}

// ______________________________________________________________________
// Testing of astencode_gen

#[cfg(test)]
fn encode_item_ast(ebml_w: &mut writer::Encoder, item: @ast::item) {
    ebml_w.start_tag(c::tag_tree as uint);
    (*item).encode(ebml_w);
    ebml_w.end_tag();
}

#[cfg(test)]
fn decode_item_ast(par_doc: ebml::Doc) -> @ast::item {
    let chi_doc = par_doc.get(c::tag_tree as uint);
    let mut d = reader::Decoder(chi_doc);
    @Decodable::decode(&mut d)
}

#[cfg(test)]
trait fake_ext_ctxt {
    fn cfg(&self) -> ast::crate_cfg;
    fn parse_sess(&self) -> @mut parse::ParseSess;
    fn call_site(&self) -> span;
    fn ident_of(&self, st: &str) -> ast::ident;
}

#[cfg(test)]
type fake_session = @mut parse::ParseSess;

#[cfg(test)]
impl fake_ext_ctxt for fake_session {
    fn cfg(&self) -> ast::crate_cfg { ~[] }
    fn parse_sess(&self) -> @mut parse::ParseSess { *self }
    fn call_site(&self) -> span {
        codemap::span {
            lo: codemap::BytePos(0),
            hi: codemap::BytePos(0),
            expn_info: None
        }
    }
    fn ident_of(&self, st: &str) -> ast::ident {
        self.interner.intern(st)
    }
}

#[cfg(test)]
fn mk_ctxt() -> @fake_ext_ctxt {
    @parse::new_parse_sess(None) as @fake_ext_ctxt
}

#[cfg(test)]
fn roundtrip(in_item: Option<@ast::item>) {
    use core::io;

    let in_item = in_item.get();
    let bytes = do io::with_bytes_writer |wr| {
        let mut ebml_w = writer::Encoder(wr);
        encode_item_ast(&mut ebml_w, in_item);
    };
    let ebml_doc = reader::Doc(@bytes);
    let out_item = decode_item_ast(ebml_doc);

    assert_eq!(in_item, out_item);
}

#[test]
fn test_basic() {
    let ext_cx = mk_ctxt();
    roundtrip(quote_item!(
        fn foo() {}
    ));
}

#[test]
fn test_smalltalk() {
    let ext_cx = mk_ctxt();
    roundtrip(quote_item!(
        fn foo() -> int { 3 + 4 } // first smalltalk program ever executed.
    ));
}

#[test]
fn test_more() {
    let ext_cx = mk_ctxt();
    roundtrip(quote_item!(
        fn foo(x: uint, y: uint) -> uint {
            let z = x + y;
            return z;
        }
    ));
}

#[test]
fn test_simplification() {
    let ext_cx = mk_ctxt();
    let item_in = ast::ii_item(quote_item!(
        fn new_int_alist<B:Copy>() -> alist<int, B> {
            fn eq_int(a: int, b: int) -> bool { a == b }
            return alist {eq_fn: eq_int, data: ~[]};
        }
    ).get());
    let item_out = simplify_ast(&item_in);
    let item_exp = ast::ii_item(quote_item!(
        fn new_int_alist<B:Copy>() -> alist<int, B> {
            return alist {eq_fn: eq_int, data: ~[]};
        }
    ).get());
    match (item_out, item_exp) {
      (ast::ii_item(item_out), ast::ii_item(item_exp)) => {
        assert!(pprust::item_to_str(item_out,
                                         ext_cx.parse_sess().interner)
                     == pprust::item_to_str(item_exp,
                                            ext_cx.parse_sess().interner));
      }
      _ => fail!()
    }
}
