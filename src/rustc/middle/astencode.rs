use util::ppaux::ty_to_str;

use syntax::ast;
use syntax::fold;
use syntax::fold::*;
use syntax::visit;
use syntax::ast_map;
use syntax::ast_util;
use syntax::codemap::span;
use std::ebml;
use std::ebml::Writer;
use std::ebml::get_doc;
use std::map::HashMap;
use std::serialization::Serializer;
use std::serialization::Deserializer;
use std::serialization::SerializerHelpers;
use std::serialization::DeserializerHelpers;
use std::prettyprint::Serializer;
use middle::{ty, typeck};
use middle::typeck::{method_origin, method_map_entry,
                        serialize_method_map_entry,
                        deserialize_method_map_entry,
                        vtable_res,
                        vtable_origin};
use driver::session::session;
use middle::freevars::{freevar_entry,
                          serialize_freevar_entry,
                          deserialize_freevar_entry};
use c = metadata::common;
use e = metadata::encoder;
use cstore = metadata::cstore;
use metadata::encoder;
use metadata::decoder;
use metadata::tyencode;
use metadata::tydecode;


// used in testing:
use syntax::diagnostic;
use syntax::codemap;
use syntax::parse;
use syntax::print::pprust;

export maps;
export encode_inlined_item;
export decode_inlined_item;

// Auxiliary maps of things to be encoded
type maps = {
    mutbl_map: middle::borrowck::mutbl_map,
    root_map: middle::borrowck::root_map,
    last_use_map: middle::liveness::last_use_map,
    method_map: middle::typeck::method_map,
    vtable_map: middle::typeck::vtable_map,
};

type decode_ctxt = @{
    cdata: cstore::crate_metadata,
    tcx: ty::ctxt,
    maps: maps
};

type extended_decode_ctxt_ = {
    dcx: decode_ctxt,
    from_id_range: ast_util::id_range,
    to_id_range: ast_util::id_range
};

enum extended_decode_ctxt {
    extended_decode_ctxt_(@extended_decode_ctxt_)
}

trait tr {
    fn tr(xcx: extended_decode_ctxt) -> self;
}

// ______________________________________________________________________
// Top-level methods.

fn encode_inlined_item(ecx: @e::encode_ctxt,
                       ebml_w: ebml::Writer,
                       path: ast_map::path,
                       ii: ast::inlined_item,
                       maps: maps) {
    debug!("> Encoding inlined item: %s::%s (%u)",
           ast_map::path_to_str(path, ecx.tcx.sess.parse_sess.interner),
           ecx.tcx.sess.str_of(ii.ident()),
           ebml_w.writer.tell());

    let id_range = ast_util::compute_id_range_for_inlined_item(ii);
    do ebml_w.wr_tag(c::tag_ast as uint) {
        ast_util::serialize_id_range(ebml_w, id_range);
        encode_ast(ebml_w, simplify_ast(ii));
        encode_side_tables_for_ii(ecx, maps, ebml_w, ii);
    }

    debug!("< Encoded inlined fn: %s::%s (%u)",
           ast_map::path_to_str(path, ecx.tcx.sess.parse_sess.interner),
           ecx.tcx.sess.str_of(ii.ident()),
           ebml_w.writer.tell());
}

fn decode_inlined_item(cdata: cstore::crate_metadata,
                       tcx: ty::ctxt,
                       maps: maps,
                       path: ast_map::path,
                       par_doc: ebml::Doc) -> Option<ast::inlined_item> {
    let dcx = @{cdata: cdata, tcx: tcx, maps: maps};
    match par_doc.opt_child(c::tag_ast) {
      None => None,
      Some(ast_doc) => {
        debug!("> Decoding inlined fn: %s::?",
               ast_map::path_to_str(path, tcx.sess.parse_sess.interner));
        let ast_dsr = ebml::ebml_deserializer(ast_doc);
        let from_id_range = ast_util::deserialize_id_range(ast_dsr);
        let to_id_range = reserve_id_range(dcx.tcx.sess, from_id_range);
        let xcx = extended_decode_ctxt_(@{dcx: dcx,
                                          from_id_range: from_id_range,
                                          to_id_range: to_id_range});
        let raw_ii = decode_ast(ast_doc);
        let ii = renumber_ast(xcx, raw_ii);
        ast_map::map_decoded_item(tcx.sess.diagnostic(),
                                  dcx.tcx.items, path, ii);
        debug!("Fn named: %s", tcx.sess.str_of(ii.ident()));
        decode_side_tables(xcx, ast_doc);
        debug!("< Decoded inlined fn: %s::%s",
               ast_map::path_to_str(path, tcx.sess.parse_sess.interner),
               tcx.sess.str_of(ii.ident()));
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

fn reserve_id_range(sess: session,
                    from_id_range: ast_util::id_range) -> ast_util::id_range {
    // Handle the case of an empty range:
    if ast_util::empty(from_id_range) { return from_id_range; }
    let cnt = from_id_range.max - from_id_range.min;
    let to_id_min = sess.parse_sess.next_id;
    let to_id_max = sess.parse_sess.next_id + cnt;
    sess.parse_sess.next_id = to_id_max;
    return {min: to_id_min, max: to_id_min};
}

impl extended_decode_ctxt {
    fn tr_id(id: ast::node_id) -> ast::node_id {
        // from_id_range should be non-empty
        assert !ast_util::empty(self.from_id_range);
        (id - self.from_id_range.min + self.to_id_range.min)
    }
    fn tr_def_id(did: ast::def_id) -> ast::def_id {
        decoder::translate_def_id(self.dcx.cdata, did)
    }
    fn tr_intern_def_id(did: ast::def_id) -> ast::def_id {
        assert did.crate == ast::local_crate;
        {crate: ast::local_crate, node: self.tr_id(did.node)}
    }
    fn tr_span(_span: span) -> span {
        ast_util::dummy_sp() // FIXME (#1972): handle span properly
    }
}

impl ast::def_id: tr {
    fn tr(xcx: extended_decode_ctxt) -> ast::def_id {
        xcx.tr_def_id(self)
    }
    fn tr_intern(xcx: extended_decode_ctxt) -> ast::def_id {
        xcx.tr_intern_def_id(self)
    }
}

impl span: tr {
    fn tr(xcx: extended_decode_ctxt) -> span {
        xcx.tr_span(self)
    }
}

trait def_id_serializer_helpers {
    fn emit_def_id(did: ast::def_id);
}

impl<S: Serializer> S: def_id_serializer_helpers {
    fn emit_def_id(did: ast::def_id) {
        ast::serialize_def_id(self, did)
    }
}

trait def_id_deserializer_helpers {
    fn read_def_id(xcx: extended_decode_ctxt) -> ast::def_id;
}

impl<D: Deserializer> D: def_id_deserializer_helpers {

    fn read_def_id(xcx: extended_decode_ctxt) -> ast::def_id {
        let did = ast::deserialize_def_id(self);
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

fn encode_ast(ebml_w: ebml::Writer, item: ast::inlined_item) {
    do ebml_w.wr_tag(c::tag_tree as uint) {
        ast::serialize_inlined_item(ebml_w, item)
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
fn simplify_ast(ii: ast::inlined_item) -> ast::inlined_item {
    fn drop_nested_items(blk: ast::blk_, fld: fold::ast_fold) -> ast::blk_ {
        let stmts_sans_items = do vec::filter(blk.stmts) |stmt| {
            match stmt.node {
              ast::stmt_expr(_, _) | ast::stmt_semi(_, _) |
              ast::stmt_decl(@{node: ast::decl_local(_), span: _}, _) => true,
              ast::stmt_decl(@{node: ast::decl_item(_), span: _}, _) => false
            }
        };
        let blk_sans_items = { stmts: stmts_sans_items,.. blk };
        fold::noop_fold_block(blk_sans_items, fld)
    }

    let fld = fold::make_fold(@{
        fold_block: fold::wrap(drop_nested_items),
        .. *fold::default_ast_fold()
    });

    match ii {
      ast::ii_item(i) => {
        ast::ii_item(fld.fold_item(i).get()) //hack: we're not dropping items
      }
      ast::ii_method(d, m) => {
        ast::ii_method(d, fld.fold_method(m))
      }
      ast::ii_foreign(i) => {
        ast::ii_foreign(fld.fold_foreign_item(i))
      }
      ast::ii_ctor(ctor, nm, tps, parent_id) => {
        let ctor_body = fld.fold_block(ctor.node.body);
        let ctor_decl = fold::fold_fn_decl(ctor.node.dec, fld);
        ast::ii_ctor({node: {body: ctor_body, dec: ctor_decl,
                              .. ctor.node},
            .. ctor}, nm, tps, parent_id)
      }
      ast::ii_dtor(dtor, nm, tps, parent_id) => {
        let dtor_body = fld.fold_block(dtor.node.body);
        ast::ii_dtor({node: {body: dtor_body,
                              .. dtor.node},
            .. dtor}, nm, tps, parent_id)
      }
    }
}

fn decode_ast(par_doc: ebml::Doc) -> ast::inlined_item {
    let chi_doc = par_doc[c::tag_tree as uint];
    let d = ebml::ebml_deserializer(chi_doc);
    ast::deserialize_inlined_item(d)
}

fn renumber_ast(xcx: extended_decode_ctxt, ii: ast::inlined_item)
    -> ast::inlined_item {
    let fld = fold::make_fold(@{
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
      ast::ii_ctor(ctor, nm, tps, parent_id) => {
        let ctor_body = fld.fold_block(ctor.node.body);
        let ctor_attrs = fld.fold_attributes(ctor.node.attrs);
        let ctor_decl = fold::fold_fn_decl(ctor.node.dec, fld);
        let new_params = fold::fold_ty_params(tps, fld);
        let ctor_id = fld.new_id(ctor.node.id);
        let new_parent = xcx.tr_def_id(parent_id);
        ast::ii_ctor({node: {body: ctor_body, attrs: ctor_attrs,
                dec: ctor_decl, id: ctor_id,
                              .. ctor.node},
            .. ctor}, nm, new_params, new_parent)
      }
      ast::ii_dtor(dtor, nm, tps, parent_id) => {
        let dtor_body = fld.fold_block(dtor.node.body);
        let dtor_attrs = fld.fold_attributes(dtor.node.attrs);
        let new_params = fold::fold_ty_params(tps, fld);
        let dtor_id = fld.new_id(dtor.node.id);
        let new_parent = xcx.tr_def_id(parent_id);
        let new_self = fld.new_id(dtor.node.self_id);
        ast::ii_dtor({node: {id: dtor_id, attrs: dtor_attrs,
                self_id: new_self, body: dtor_body},
                        .. dtor},
          nm, new_params, new_parent)
      }
     }
}

// ______________________________________________________________________
// Encoding and decoding of ast::def

fn encode_def(ebml_w: ebml::Writer, def: ast::def) {
    ast::serialize_def(ebml_w, def)
}

fn decode_def(xcx: extended_decode_ctxt, doc: ebml::Doc) -> ast::def {
    let dsr = ebml::ebml_deserializer(doc);
    let def = ast::deserialize_def(dsr);
    def.tr(xcx)
}

impl ast::def: tr {
    fn tr(xcx: extended_decode_ctxt) -> ast::def {
        match self {
          ast::def_fn(did, p) => { ast::def_fn(did.tr(xcx), p) }
          ast::def_static_method(did, p) => {
            ast::def_static_method(did.tr(xcx), p)
          }
          ast::def_self(nid) => { ast::def_self(xcx.tr_id(nid)) }
          ast::def_mod(did) => { ast::def_mod(did.tr(xcx)) }
          ast::def_foreign_mod(did) => { ast::def_foreign_mod(did.tr(xcx)) }
          ast::def_const(did) => { ast::def_const(did.tr(xcx)) }
          ast::def_arg(nid, m) => { ast::def_arg(xcx.tr_id(nid), m) }
          ast::def_local(nid, b) => { ast::def_local(xcx.tr_id(nid), b) }
          ast::def_variant(e_did, v_did) => {
            ast::def_variant(e_did.tr(xcx), v_did.tr(xcx))
          }
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
          ast::def_class(did, has_constructor) => {
            ast::def_class(did.tr(xcx), has_constructor)
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
// Encoding and decoding of freevar information

fn encode_freevar_entry(ebml_w: ebml::Writer, fv: freevar_entry) {
    serialize_freevar_entry(ebml_w, fv)
}

trait ebml_deserializer_helper {
    fn read_freevar_entry(xcx: extended_decode_ctxt) -> freevar_entry;
}

impl ebml::EbmlDeserializer: ebml_deserializer_helper {
    fn read_freevar_entry(xcx: extended_decode_ctxt) -> freevar_entry {
        let fv = deserialize_freevar_entry(self);
        fv.tr(xcx)
    }
}

impl freevar_entry: tr {
    fn tr(xcx: extended_decode_ctxt) -> freevar_entry {
        {def: self.def.tr(xcx), span: self.span.tr(xcx)}
    }
}

// ______________________________________________________________________
// Encoding and decoding of method_map_entry

trait read_method_map_entry_helper {
    fn read_method_map_entry(xcx: extended_decode_ctxt) -> method_map_entry;
}

impl ebml::EbmlDeserializer: read_method_map_entry_helper {
    fn read_method_map_entry(xcx: extended_decode_ctxt) -> method_map_entry {
        let mme = deserialize_method_map_entry(self);
        {derefs: mme.derefs,
         self_mode: mme.self_mode,
         origin: mme.origin.tr(xcx)}
    }
}

impl method_origin: tr {
    fn tr(xcx: extended_decode_ctxt) -> method_origin {
        match self {
          typeck::method_static(did) => {
            typeck::method_static(did.tr(xcx))
          }
          typeck::method_param(mp) => {
            typeck::method_param({trait_id:mp.trait_id.tr(xcx),.. mp})
          }
          typeck::method_trait(did, m) => {
            typeck::method_trait(did.tr(xcx), m)
          }
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding vtable_res

fn encode_vtable_res(ecx: @e::encode_ctxt,
                   ebml_w: ebml::Writer,
                   dr: typeck::vtable_res) {
    // can't autogenerate this code because automatic serialization of
    // ty::t doesn't work, and there is no way (atm) to have
    // hand-written serialization routines combine with auto-generated
    // ones.  perhaps we should fix this.
    do ebml_w.emit_from_vec(*dr) |vtable_origin| {
        encode_vtable_origin(ecx, ebml_w, vtable_origin)
    }
}

fn encode_vtable_origin(ecx: @e::encode_ctxt,
                      ebml_w: ebml::Writer,
                      vtable_origin: typeck::vtable_origin) {
    do ebml_w.emit_enum(~"vtable_origin") {
        match vtable_origin {
          typeck::vtable_static(def_id, tys, vtable_res) => {
            do ebml_w.emit_enum_variant(~"vtable_static", 0u, 3u) {
                do ebml_w.emit_enum_variant_arg(0u) {
                    ebml_w.emit_def_id(def_id)
                }
                do ebml_w.emit_enum_variant_arg(1u) {
                    ebml_w.emit_tys(ecx, tys);
                }
                do ebml_w.emit_enum_variant_arg(2u) {
                    encode_vtable_res(ecx, ebml_w, vtable_res);
                }
            }
          }
          typeck::vtable_param(pn, bn) => {
            do ebml_w.emit_enum_variant(~"vtable_param", 1u, 2u) {
                do ebml_w.emit_enum_variant_arg(0u) {
                    ebml_w.emit_uint(pn);
                }
                do ebml_w.emit_enum_variant_arg(1u) {
                    ebml_w.emit_uint(bn);
                }
            }
          }
          typeck::vtable_trait(def_id, tys) => {
            do ebml_w.emit_enum_variant(~"vtable_trait", 1u, 3u) {
                do ebml_w.emit_enum_variant_arg(0u) {
                    ebml_w.emit_def_id(def_id)
                }
                do ebml_w.emit_enum_variant_arg(1u) {
                    ebml_w.emit_tys(ecx, tys);
                }
            }
          }
        }
    }

}

trait vtable_deserialization_helpers {
    fn read_vtable_res(xcx: extended_decode_ctxt) -> typeck::vtable_res;
    fn read_vtable_origin(xcx: extended_decode_ctxt) -> typeck::vtable_origin;
}

impl ebml::EbmlDeserializer: vtable_deserialization_helpers {
    fn read_vtable_res(xcx: extended_decode_ctxt) -> typeck::vtable_res {
        @self.read_to_vec(|| self.read_vtable_origin(xcx) )
    }

    fn read_vtable_origin(xcx: extended_decode_ctxt)
        -> typeck::vtable_origin {
        do self.read_enum(~"vtable_origin") {
            do self.read_enum_variant |i| {
                match i {
                  0 => {
                    typeck::vtable_static(
                        do self.read_enum_variant_arg(0u) {
                            self.read_def_id(xcx)
                        },
                        do self.read_enum_variant_arg(1u) {
                            self.read_tys(xcx)
                        },
                        do self.read_enum_variant_arg(2u) {
                            self.read_vtable_res(xcx)
                        }
                    )
                  }
                  1 => {
                    typeck::vtable_param(
                        do self.read_enum_variant_arg(0u) {
                            self.read_uint()
                        },
                        do self.read_enum_variant_arg(1u) {
                            self.read_uint()
                        }
                    )
                  }
                  2 => {
                    typeck::vtable_trait(
                        do self.read_enum_variant_arg(0u) {
                            self.read_def_id(xcx)
                        },
                        do self.read_enum_variant_arg(1u) {
                            self.read_tys(xcx)
                        }
                    )
                  }
                  // hard to avoid - user input
                  _ => fail ~"bad enum variant"
                }
            }
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding the side tables

trait get_ty_str_ctxt {
    fn ty_str_ctxt() -> @tyencode::ctxt;
}

impl @e::encode_ctxt: get_ty_str_ctxt {
    fn ty_str_ctxt() -> @tyencode::ctxt {
        @{diag: self.tcx.sess.diagnostic(),
          ds: e::def_to_str,
          tcx: self.tcx,
          reachable: |a| encoder::reachable(self, a),
          abbrevs: tyencode::ac_use_abbrevs(self.type_abbrevs)}
    }
}

trait ebml_writer_helpers {
    fn emit_ty(ecx: @e::encode_ctxt, ty: ty::t);
    fn emit_tys(ecx: @e::encode_ctxt, tys: ~[ty::t]);
    fn emit_bounds(ecx: @e::encode_ctxt, bs: ty::param_bounds);
    fn emit_tpbt(ecx: @e::encode_ctxt, tpbt: ty::ty_param_bounds_and_ty);
}

impl ebml::Writer: ebml_writer_helpers {
    fn emit_ty(ecx: @e::encode_ctxt, ty: ty::t) {
        e::write_type(ecx, self, ty)
    }

    fn emit_tys(ecx: @e::encode_ctxt, tys: ~[ty::t]) {
        do self.emit_from_vec(tys) |ty| {
            e::write_type(ecx, self, ty)
        }
    }

    fn emit_bounds(ecx: @e::encode_ctxt, bs: ty::param_bounds) {
        tyencode::enc_bounds(self.writer, ecx.ty_str_ctxt(), bs)
    }

    fn emit_tpbt(ecx: @e::encode_ctxt, tpbt: ty::ty_param_bounds_and_ty) {
        do self.emit_rec {
            do self.emit_rec_field(~"bounds", 0u) {
                do self.emit_from_vec(*tpbt.bounds) |bs| {
                    self.emit_bounds(ecx, bs);
                }
            }
            do self.emit_rec_field(~"region_param", 1u) {
                ty::serialize_opt_region_variance(
                    self,
                    tpbt.region_param);
            }
            do self.emit_rec_field(~"ty", 2u) {
                self.emit_ty(ecx, tpbt.ty);
            }
        }
    }
}

trait write_tag_and_id {
    fn tag(tag_id: c::astencode_tag, f: fn());
    fn id(id: ast::node_id);
}

impl ebml::Writer: write_tag_and_id {
    fn tag(tag_id: c::astencode_tag, f: fn()) {
        do self.wr_tag(tag_id as uint) { f() }
    }

    fn id(id: ast::node_id) {
        self.wr_tagged_u64(c::tag_table_id as uint, id as u64)
    }
}

fn encode_side_tables_for_ii(ecx: @e::encode_ctxt,
                             maps: maps,
                             ebml_w: ebml::Writer,
                             ii: ast::inlined_item) {
    do ebml_w.wr_tag(c::tag_table as uint) {
        ast_util::visit_ids_for_inlined_item(
            ii,
            fn@(id: ast::node_id, copy ebml_w) {
                // Note: this will cause a copy of ebml_w, which is bad as
                // it has mut fields.  But I believe it's harmless since
                // we generate balanced EBML.
                encode_side_tables_for_id(ecx, maps, ebml_w, id)
            });
    }
}

fn encode_side_tables_for_id(ecx: @e::encode_ctxt,
                             maps: maps,
                             ebml_w: ebml::Writer,
                             id: ast::node_id) {
    let tcx = ecx.tcx;

    debug!("Encoding side tables for id %d", id);

    do option::iter(tcx.def_map.find(id)) |def| {
        do ebml_w.tag(c::tag_table_def) {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) {
                ast::serialize_def(ebml_w, def)
            }
        }
    }
    do option::iter((*tcx.node_types).find(id as uint)) |ty| {
        do ebml_w.tag(c::tag_table_node_type) {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) {
                e::write_type(ecx, ebml_w, ty)
            }
        }
    }

    do option::iter(tcx.node_type_substs.find(id)) |tys| {
        do ebml_w.tag(c::tag_table_node_type_subst) {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) {
                ebml_w.emit_tys(ecx, tys)
            }
        }
    }

    do option::iter(tcx.freevars.find(id)) |fv| {
        do ebml_w.tag(c::tag_table_freevars) {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) {
                do ebml_w.emit_from_vec(*fv) |fv_entry| {
                    encode_freevar_entry(ebml_w, *fv_entry)
                }
            }
        }
    }

    let lid = {crate: ast::local_crate, node: id};
    do option::iter(tcx.tcache.find(lid)) |tpbt| {
        do ebml_w.tag(c::tag_table_tcache) {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) {
                ebml_w.emit_tpbt(ecx, tpbt);
            }
        }
    }

    do option::iter(tcx.ty_param_bounds.find(id)) |pbs| {
        do ebml_w.tag(c::tag_table_param_bounds) {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) {
                ebml_w.emit_bounds(ecx, pbs)
            }
        }
    }

    // I believe it is not necessary to encode this information.  The
    // ids will appear in the AST but in the *type* information, which
    // is what we actually use in trans, all modes will have been
    // resolved.
    //
    //option::iter(tcx.inferred_modes.find(id)) {|m|
    //    ebml_w.tag(c::tag_table_inferred_modes) {||
    //        ebml_w.id(id);
    //        ebml_w.tag(c::tag_table_val) {||
    //            tyencode::enc_mode(ebml_w.writer, ty_str_ctxt(), m);
    //        }
    //    }
    //}

    do option::iter(maps.mutbl_map.find(id)) |_m| {
        do ebml_w.tag(c::tag_table_mutbl) {
            ebml_w.id(id);
        }
    }

    do option::iter(maps.last_use_map.find(id)) |m| {
        do ebml_w.tag(c::tag_table_last_use) {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) {
                do ebml_w.emit_from_vec((*m).get()) |id| {
                    ebml_w.emit_int(id);
                }
            }
        }
    }

    do option::iter(maps.method_map.find(id)) |mme| {
        do ebml_w.tag(c::tag_table_method_map) {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) {
                serialize_method_map_entry(ebml_w, mme)
            }
        }
    }

    do option::iter(maps.vtable_map.find(id)) |dr| {
        do ebml_w.tag(c::tag_table_vtable_map) {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) {
                encode_vtable_res(ecx, ebml_w, dr);
            }
        }
    }

    do option::iter(tcx.borrowings.find(id)) |_borrow| {
        do ebml_w.tag(c::tag_table_borrowings) {
            ebml_w.id(id);
            do ebml_w.tag(c::tag_table_val) {
                // N.B. We don't actually serialize borrows as, in
                // trans, we only care whether a value is borrowed or
                // not.
            }
        }
    }
}

trait doc_decoder_helpers {
    fn as_int() -> int;
    fn opt_child(tag: c::astencode_tag) -> Option<ebml::Doc>;
}

impl ebml::Doc: doc_decoder_helpers {
    fn as_int() -> int { ebml::doc_as_u64(self) as int }
    fn opt_child(tag: c::astencode_tag) -> Option<ebml::Doc> {
        ebml::maybe_get_doc(self, tag as uint)
    }
}

trait ebml_deserializer_decoder_helpers {
    fn read_ty(xcx: extended_decode_ctxt) -> ty::t;
    fn read_tys(xcx: extended_decode_ctxt) -> ~[ty::t];
    fn read_bounds(xcx: extended_decode_ctxt) -> @~[ty::param_bound];
    fn read_ty_param_bounds_and_ty(xcx: extended_decode_ctxt)
                                -> ty::ty_param_bounds_and_ty;
}

impl ebml::EbmlDeserializer: ebml_deserializer_decoder_helpers {

    fn read_ty(xcx: extended_decode_ctxt) -> ty::t {
        // Note: regions types embed local node ids.  In principle, we
        // should translate these node ids into the new decode
        // context.  However, we do not bother, because region types
        // are not used during trans.

        tydecode::parse_ty_data(
            self.parent.data, xcx.dcx.cdata.cnum, self.pos, xcx.dcx.tcx,
            |a| xcx.tr_def_id(a) )
    }

    fn read_tys(xcx: extended_decode_ctxt) -> ~[ty::t] {
        self.read_to_vec(|| self.read_ty(xcx) )
    }

    fn read_bounds(xcx: extended_decode_ctxt) -> @~[ty::param_bound] {
        tydecode::parse_bounds_data(
            self.parent.data, self.pos, xcx.dcx.cdata.cnum, xcx.dcx.tcx,
            |a| xcx.tr_def_id(a) )
    }

    fn read_ty_param_bounds_and_ty(xcx: extended_decode_ctxt)
        -> ty::ty_param_bounds_and_ty {
        do self.read_rec {
            {
                bounds: self.read_rec_field(~"bounds", 0u, || {
                    @self.read_to_vec(|| self.read_bounds(xcx) )
                }),
                region_param: self.read_rec_field(~"region_param", 1u, || {
                    ty::deserialize_opt_region_variance(self)
                }),
                ty: self.read_rec_field(~"ty", 2u, || {
                    self.read_ty(xcx)
                })
            }
        }
    }
}

fn decode_side_tables(xcx: extended_decode_ctxt,
                      ast_doc: ebml::Doc) {
    let dcx = xcx.dcx;
    let tbl_doc = ast_doc[c::tag_table as uint];
    for ebml::docs(tbl_doc) |tag, entry_doc| {
        let id0 = entry_doc[c::tag_table_id as uint].as_int();
        let id = xcx.tr_id(id0);

        debug!(">> Side table document with tag 0x%x \
                found for id %d (orig %d)",
               tag, id, id0);

        if tag == (c::tag_table_mutbl as uint) {
            dcx.maps.mutbl_map.insert(id, ());
        } else {
            let val_doc = entry_doc[c::tag_table_val as uint];
            let val_dsr = ebml::ebml_deserializer(val_doc);
            if tag == (c::tag_table_def as uint) {
                let def = decode_def(xcx, val_doc);
                dcx.tcx.def_map.insert(id, def);
            } else if tag == (c::tag_table_node_type as uint) {
                let ty = val_dsr.read_ty(xcx);
                (*dcx.tcx.node_types).insert(id as uint, ty);
            } else if tag == (c::tag_table_node_type_subst as uint) {
                let tys = val_dsr.read_tys(xcx);
                dcx.tcx.node_type_substs.insert(id, tys);
            } else if tag == (c::tag_table_freevars as uint) {
                let fv_info = @val_dsr.read_to_vec(|| {
                    @val_dsr.read_freevar_entry(xcx)
                });
                dcx.tcx.freevars.insert(id, fv_info);
            } else if tag == (c::tag_table_tcache as uint) {
                let tpbt = val_dsr.read_ty_param_bounds_and_ty(xcx);
                let lid = {crate: ast::local_crate, node: id};
                dcx.tcx.tcache.insert(lid, tpbt);
            } else if tag == (c::tag_table_param_bounds as uint) {
                let bounds = val_dsr.read_bounds(xcx);
                dcx.tcx.ty_param_bounds.insert(id, bounds);
            } else if tag == (c::tag_table_last_use as uint) {
                let ids = val_dsr.read_to_vec(|| {
                    xcx.tr_id(val_dsr.read_int())
                });
                let dvec = @dvec::from_vec(vec::to_mut(ids));
                dcx.maps.last_use_map.insert(id, dvec);
            } else if tag == (c::tag_table_method_map as uint) {
                dcx.maps.method_map.insert(
                    id,
                    val_dsr.read_method_map_entry(xcx));
            } else if tag == (c::tag_table_vtable_map as uint) {
                dcx.maps.vtable_map.insert(id,
                                           val_dsr.read_vtable_res(xcx));
            } else if tag == (c::tag_table_borrowings as uint) {
                // N.B.: we don't actually *serialize* borrows because, in
                // trans, the only thing we care about is whether a value was
                // borrowed or not.
                let borrow = {region: ty::re_static, mutbl: ast::m_imm};
                dcx.tcx.borrowings.insert(id, borrow);
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
fn encode_item_ast(ebml_w: ebml::Writer, item: @ast::item) {
    do ebml_w.wr_tag(c::tag_tree as uint) {
        ast::serialize_item(ebml_w, *item);
    }
}

#[cfg(test)]
fn decode_item_ast(par_doc: ebml::Doc) -> @ast::item {
    let chi_doc = par_doc[c::tag_tree as uint];
    let d = ebml::ebml_deserializer(chi_doc);
    @ast::deserialize_item(d)
}

#[cfg(test)]
trait fake_ext_ctxt {
    fn cfg() -> ast::crate_cfg;
    fn parse_sess() -> parse::parse_sess;
}

#[cfg(test)]
type fake_session = parse::parse_sess;

#[cfg(test)]
impl fake_session: fake_ext_ctxt {
    fn cfg() -> ast::crate_cfg { ~[] }
    fn parse_sess() -> parse::parse_sess { self }
}

#[cfg(test)]
fn mk_ctxt() -> fake_ext_ctxt {
    parse::new_parse_sess(None) as fake_ext_ctxt
}

#[cfg(test)]
fn roundtrip(in_item: @ast::item) {
    let mbuf = io::mem_buffer();
    let ebml_w = ebml::Writer(io::mem_buffer_writer(mbuf));
    encode_item_ast(ebml_w, in_item);
    let ebml_doc = ebml::Doc(@io::mem_buffer_buf(mbuf));
    let out_item = decode_item_ast(ebml_doc);

    let exp_str =
        io::with_str_writer(|w| ast::serialize_item(w, *in_item) );
    let out_str =
        io::with_str_writer(|w| ast::serialize_item(w, *out_item) );

    debug!("expected string: %s", exp_str);
    debug!("actual string  : %s", out_str);

    assert exp_str == out_str;
}

#[test]
fn test_basic() {
    let ext_cx = mk_ctxt();
    roundtrip(#ast[item]{
        fn foo() {}
    });
}

#[test]
fn test_smalltalk() {
    let ext_cx = mk_ctxt();
    roundtrip(#ast[item]{
        fn foo() -> int { 3 + 4 } // first smalltalk program ever executed.
    });
}

#[test]
fn test_more() {
    let ext_cx = mk_ctxt();
    roundtrip(#ast[item]{
        fn foo(x: uint, y: uint) -> uint {
            let z = x + y;
            return z;
        }
    });
}

#[test]
fn test_simplification() {
    let ext_cx = mk_ctxt();
    let item_in = ast::ii_item(#ast[item] {
        fn new_int_alist<B: Copy>() -> alist<int, B> {
            fn eq_int(&&a: int, &&b: int) -> bool { a == b }
            return {eq_fn: eq_int, mut data: ~[]};
        }
    });
    let item_out = simplify_ast(item_in);
    let item_exp = ast::ii_item(#ast[item] {
        fn new_int_alist<B: Copy>() -> alist<int, B> {
            return {eq_fn: eq_int, mut data: ~[]};
        }
    });
    match (item_out, item_exp) {
      (ast::ii_item(item_out), ast::ii_item(item_exp)) => {
        assert pprust::item_to_str(item_out, ext_cx.parse_sess().interner)
            == pprust::item_to_str(item_exp, ext_cx.parse_sess().interner);
      }
      _ => fail
    }
}
