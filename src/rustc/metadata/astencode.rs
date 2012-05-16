import util::ppaux::ty_to_str;

import syntax::ast;
import syntax::ast_util::{id_range, compute_id_range_for_inlined_item,
                          visit_ids_for_inlined_item, serialize_id_range,
                          deserialize_id_range};
import syntax::fold;
import syntax::visit;
import syntax::ast_util;
import syntax::ast_util::inlined_item_methods;
import syntax::codemap::span;
import std::ebml;
import std::ebml::writer;
import std::ebml::serializer;
import std::ebml::deserializer;
import std::map::hashmap;
import std::serialization::serializer;
import std::serialization::deserializer;
import std::serialization::serializer_helpers;
import std::serialization::deserializer_helpers;
import std::prettyprint::serializer;
import std::smallintmap::map;
import middle::trans::common::maps;
import middle::{ty, typeck, last_use, ast_map};
import middle::typeck::{method_origin,
                        serialize_method_origin,
                        deserialize_method_origin,
                        vtable_res,
                        vtable_origin};
import driver::session::session;
import middle::freevars::{freevar_entry,
                          serialize_freevar_entry,
                          deserialize_freevar_entry};
import c = common;
import e = encoder;

// used in testing:
import driver::diagnostic;
import syntax::codemap;
import syntax::parse;
import syntax::print::pprust;

export encode_inlined_item;
export decode_inlined_item;

type decode_ctxt = @{
    cdata: cstore::crate_metadata,
    tcx: ty::ctxt,
    maps: maps
};

type extended_decode_ctxt = @{
    dcx: decode_ctxt,
    from_id_range: id_range,
    to_id_range: id_range
};

iface tr {
    fn tr(xcx: extended_decode_ctxt) -> self;
}

// ______________________________________________________________________
// Top-level methods.

fn encode_inlined_item(ecx: @e::encode_ctxt,
                       ebml_w: ebml::writer,
                       path: ast_map::path,
                       ii: ast::inlined_item) {
    #debug["> Encoding inlined item: %s::%s (%u)",
           ast_map::path_to_str(path), ii.ident(),
           ebml_w.writer.tell()];

    let id_range = compute_id_range_for_inlined_item(ii);
    ebml_w.wr_tag(c::tag_ast as uint) {||
        encode_id_range(ebml_w, id_range);
        encode_ast(ebml_w, simplify_ast(ii));
        encode_side_tables_for_ii(ecx, ebml_w, ii);
    }

    #debug["< Encoded inlined fn: %s::%s (%u)",
           ast_map::path_to_str(path), ii.ident(),
           ebml_w.writer.tell()];
}

fn decode_inlined_item(cdata: cstore::crate_metadata,
                       tcx: ty::ctxt,
                       maps: maps,
                       path: ast_map::path,
                       par_doc: ebml::doc) -> option<ast::inlined_item> {
    let dcx = @{cdata: cdata, tcx: tcx, maps: maps};
    alt par_doc.opt_child(c::tag_ast) {
      none { none }
      some(ast_doc) {
        #debug["> Decoding inlined fn: %s::?", ast_map::path_to_str(path)];
        let from_id_range = decode_id_range(ast_doc);
        let to_id_range = reserve_id_range(dcx.tcx.sess, from_id_range);
        let xcx = @{dcx: dcx,
                    from_id_range: from_id_range,
                    to_id_range: to_id_range};
        let raw_ii = decode_ast(ast_doc);
        let ii = renumber_ast(xcx, raw_ii);
        ast_map::map_decoded_item(tcx.sess, dcx.tcx.items, path, ii);
        #debug["Fn named: %s", ii.ident()];
        decode_side_tables(xcx, ast_doc);
        #debug["< Decoded inlined fn: %s::%s",
               ast_map::path_to_str(path), ii.ident()];
        alt ii {
          ast::ii_item(i) {
            #debug(">>> DECODED ITEM >>>\n%s\n<<< DECODED ITEM <<<",
                   rustsyntax::print::pprust::item_to_str(i));
          }
          _ { }
        }
        some(ii)
      }
    }
}

// ______________________________________________________________________
// Enumerating the IDs which appear in an AST

fn encode_id_range(ebml_w: ebml::writer, id_range: id_range) {
    serialize_id_range(ebml_w, id_range);
}

fn decode_id_range(par_doc: ebml::doc) -> id_range {
    let range_doc = par_doc[c::tag_id_range];
    let dsr = ebml::ebml_deserializer(range_doc);
    deserialize_id_range(dsr)
}

fn reserve_id_range(sess: session, from_id_range: id_range) -> id_range {
    // Handle the case of an empty range:
    if empty(from_id_range) { ret from_id_range; }
    let cnt = from_id_range.max - from_id_range.min;
    let to_id_min = sess.parse_sess.next_id;
    let to_id_max = sess.parse_sess.next_id + cnt;
    sess.parse_sess.next_id = to_id_max;
    ret {min: to_id_min, max: to_id_min};
}

impl translation_routines for extended_decode_ctxt {
    fn tr_id(id: ast::node_id) -> ast::node_id {
        // from_id_range should be non-empty
        assert !empty(self.from_id_range);
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
        ast_util::dummy_sp() // TODO...
    }
}

impl of tr for ast::def_id {
    fn tr(xcx: extended_decode_ctxt) -> ast::def_id {
        xcx.tr_def_id(self)
    }
    fn tr_intern(xcx: extended_decode_ctxt) -> ast::def_id {
        xcx.tr_intern_def_id(self)
    }
}

impl of tr for span {
    fn tr(xcx: extended_decode_ctxt) -> span {
        xcx.tr_span(self)
    }
}

impl serializer_helpers<S: serializer> for S {
    fn emit_def_id(did: ast::def_id) {
        ast::serialize_def_id(self, did)
    }
}

impl deserializer_helpers<D: deserializer> for D {
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

fn encode_ast(ebml_w: ebml::writer, item: ast::inlined_item) {
    ebml_w.wr_tag(c::tag_tree as uint) {||
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
        let stmts_sans_items = vec::filter(blk.stmts) {|stmt|
            alt stmt.node {
              ast::stmt_expr(_, _) | ast::stmt_semi(_, _) |
              ast::stmt_decl(@{node: ast::decl_local(_), span: _}, _) { true }
              ast::stmt_decl(@{node: ast::decl_item(_), span: _}, _) { false }
            }
        };
        let blk_sans_items = { stmts: stmts_sans_items with blk };
        fold::noop_fold_block(blk_sans_items, fld)
    }

    let fld = fold::make_fold({
        fold_block: fold::wrap(drop_nested_items)
        with *fold::default_ast_fold()
    });

    alt ii {
      ast::ii_item(i) {
        ast::ii_item(fld.fold_item(i))
      }
      ast::ii_method(d, m) {
        ast::ii_method(d, fld.fold_method(m))
      }
      ast::ii_native(i) {
        ast::ii_native(fld.fold_native_item(i))
      }
      ast::ii_ctor(ctor, nm, tps, parent_id) {
        let ctor_body = fld.fold_block(ctor.node.body);
        let ctor_decl = fold::fold_fn_decl(ctor.node.dec, fld);
        ast::ii_ctor({node: {body: ctor_body, dec: ctor_decl
                              with ctor.node}
            with ctor}, nm, tps, parent_id)
      }
    }
}

fn decode_ast(par_doc: ebml::doc) -> ast::inlined_item {
    let chi_doc = par_doc[c::tag_tree];
    let d = ebml::ebml_deserializer(chi_doc);
    ast::deserialize_inlined_item(d)
}

fn renumber_ast(xcx: extended_decode_ctxt, ii: ast::inlined_item)
    -> ast::inlined_item {
    let fld = fold::make_fold({
        new_id: xcx.tr_id(_),
        new_span: xcx.tr_span(_)
        with *fold::default_ast_fold()
    });

    alt ii {
      ast::ii_item(i) {
        ast::ii_item(fld.fold_item(i))
      }
      ast::ii_method(d, m) {
        ast::ii_method(xcx.tr_def_id(d), fld.fold_method(m))
      }
      ast::ii_native(i) {
        ast::ii_native(fld.fold_native_item(i))
      }
      ast::ii_ctor(ctor, nm, tps, parent_id) {
        let ctor_body = fld.fold_block(ctor.node.body);
        let ctor_decl = fold::fold_fn_decl(ctor.node.dec, fld);
        let new_params = fold::fold_ty_params(tps, fld);
        let ctor_id = fld.new_id(ctor.node.id);
        let new_parent = xcx.tr_def_id(parent_id);
        ast::ii_ctor({node: {body: ctor_body, dec: ctor_decl, id: ctor_id
                              with ctor.node}
            with ctor}, nm, new_params, new_parent)
      }
     }
}

// ______________________________________________________________________
// Encoding and decoding of ast::def

fn encode_def(ebml_w: ebml::writer, def: ast::def) {
    ast::serialize_def(ebml_w, def)
}

fn decode_def(xcx: extended_decode_ctxt, doc: ebml::doc) -> ast::def {
    let dsr = ebml::ebml_deserializer(doc);
    let def = ast::deserialize_def(dsr);
    def.tr(xcx)
}

impl of tr for ast::def {
    fn tr(xcx: extended_decode_ctxt) -> ast::def {
        alt self {
          ast::def_fn(did, p) { ast::def_fn(did.tr(xcx), p) }
          ast::def_self(nid) { ast::def_self(xcx.tr_id(nid)) }
          ast::def_mod(did) { ast::def_mod(did.tr(xcx)) }
          ast::def_native_mod(did) { ast::def_native_mod(did.tr(xcx)) }
          ast::def_const(did) { ast::def_const(did.tr(xcx)) }
          ast::def_arg(nid, m) { ast::def_arg(xcx.tr_id(nid), m) }
          ast::def_local(nid, b) { ast::def_local(xcx.tr_id(nid), b) }
          ast::def_variant(e_did, v_did) {
            ast::def_variant(e_did.tr(xcx), v_did.tr(xcx))
          }
          ast::def_ty(did) { ast::def_ty(did.tr(xcx)) }
          ast::def_prim_ty(p) { ast::def_prim_ty(p) }
          ast::def_ty_param(did, v) { ast::def_ty_param(did.tr(xcx), v) }
          ast::def_binding(nid) { ast::def_binding(xcx.tr_id(nid)) }
          ast::def_use(did) { ast::def_use(did.tr(xcx)) }
          ast::def_upvar(nid1, def, nid2) {
            ast::def_upvar(xcx.tr_id(nid1), @(*def).tr(xcx), xcx.tr_id(nid2))
          }
          ast::def_class(did) {
            ast::def_class(did.tr(xcx))
          }
          ast::def_region(nid) { ast::def_region(xcx.tr_id(nid)) }
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of freevar information

fn encode_freevar_entry(ebml_w: ebml::writer, fv: freevar_entry) {
    serialize_freevar_entry(ebml_w, fv)
}

impl helper for ebml::ebml_deserializer {
    fn read_freevar_entry(xcx: extended_decode_ctxt) -> freevar_entry {
        let fv = deserialize_freevar_entry(self);
        fv.tr(xcx)
    }
}

impl of tr for freevar_entry {
    fn tr(xcx: extended_decode_ctxt) -> freevar_entry {
        {def: self.def.tr(xcx), span: self.span.tr(xcx)}
    }
}

// ______________________________________________________________________
// Encoding and decoding of method_origin

fn encode_method_origin(ebml_w: ebml::writer, mo: method_origin) {
    serialize_method_origin(ebml_w, mo)
}

impl helper for ebml::ebml_deserializer {
    fn read_method_origin(xcx: extended_decode_ctxt) -> method_origin {
        let fv = deserialize_method_origin(self);
        fv.tr(xcx)
    }
    fn read_is_last_use(xcx: extended_decode_ctxt) -> last_use::is_last_use {
        let lu = last_use::deserialize_is_last_use(self);
        lu.tr(xcx)
    }
}

impl of tr for method_origin {
    fn tr(xcx: extended_decode_ctxt) -> method_origin {
        alt self {
          typeck::method_static(did) {
            typeck::method_static(did.tr(xcx))
          }
          typeck::method_param(did, m, p, b) {
            typeck::method_param(did.tr(xcx), m, p, b)
          }
          typeck::method_iface(did, m) {
            typeck::method_iface(did.tr(xcx), m)
          }
        }
    }
}

impl of tr for last_use::is_last_use {
    fn tr(xcx: extended_decode_ctxt) -> last_use::is_last_use {
        alt self {
          last_use::is_last_use { self }
          last_use::closes_over(ids) {
            last_use::closes_over(vec::map(ids, {|id| xcx.tr_id(id)}))
          }
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding vtable_res

fn encode_vtable_res(ecx: @e::encode_ctxt,
                   ebml_w: ebml::writer,
                   dr: typeck::vtable_res) {
    // can't autogenerate this code because automatic serialization of
    // ty::t doesn't work, and there is no way (atm) to have
    // hand-written serialization routines combine with auto-generated
    // ones.  perhaps we should fix this.
    ebml_w.emit_from_vec(*dr) {|vtable_origin|
        encode_vtable_origin(ecx, ebml_w, vtable_origin)
    }
}

fn encode_vtable_origin(ecx: @e::encode_ctxt,
                      ebml_w: ebml::writer,
                      vtable_origin: typeck::vtable_origin) {
    ebml_w.emit_enum("vtable_origin") {||
        alt vtable_origin {
          typeck::vtable_static(def_id, tys, vtable_res) {
            ebml_w.emit_enum_variant("vtable_static", 0u, 3u) {||
                ebml_w.emit_enum_variant_arg(0u) {||
                    ebml_w.emit_def_id(def_id)
                }
                ebml_w.emit_enum_variant_arg(1u) {||
                    ebml_w.emit_tys(ecx, tys);
                }
                ebml_w.emit_enum_variant_arg(2u) {||
                    encode_vtable_res(ecx, ebml_w, vtable_res);
                }
            }
          }
          typeck::vtable_param(pn, bn) {
            ebml_w.emit_enum_variant("vtable_param", 1u, 2u) {||
                ebml_w.emit_enum_variant_arg(0u) {||
                    ebml_w.emit_uint(pn);
                }
                ebml_w.emit_enum_variant_arg(1u) {||
                    ebml_w.emit_uint(bn);
                }
            }
          }
          typeck::vtable_iface(def_id, tys) {
            ebml_w.emit_enum_variant("vtable_iface", 1u, 3u) {||
                ebml_w.emit_enum_variant_arg(0u) {||
                    ebml_w.emit_def_id(def_id)
                }
                ebml_w.emit_enum_variant_arg(1u) {||
                    ebml_w.emit_tys(ecx, tys);
                }
            }
          }
        }
    }

}

impl helpers for ebml::ebml_deserializer {
    fn read_vtable_res(xcx: extended_decode_ctxt) -> typeck::vtable_res {
        @self.read_to_vec {|| self.read_vtable_origin(xcx) }
    }

    fn read_vtable_origin(xcx: extended_decode_ctxt)
        -> typeck::vtable_origin {
        self.read_enum("vtable_origin") {||
            self.read_enum_variant {|i|
                alt check i {
                  0u {
                    typeck::vtable_static(
                        self.read_enum_variant_arg(0u) {||
                            self.read_def_id(xcx)
                        },
                        self.read_enum_variant_arg(1u) {||
                            self.read_tys(xcx)
                        },
                        self.read_enum_variant_arg(2u) {||
                            self.read_vtable_res(xcx)
                        }
                    )
                  }
                  1u {
                    typeck::vtable_param(
                        self.read_enum_variant_arg(0u) {||
                            self.read_uint()
                        },
                        self.read_enum_variant_arg(1u) {||
                            self.read_uint()
                        }
                    )
                  }
                  2u {
                    typeck::vtable_iface(
                        self.read_enum_variant_arg(0u) {||
                            self.read_def_id(xcx)
                        },
                        self.read_enum_variant_arg(1u) {||
                            self.read_tys(xcx)
                        }
                    )
                  }
                }
            }
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding the side tables

impl helpers for @e::encode_ctxt {
    fn ty_str_ctxt() -> @tyencode::ctxt {
        @{ds: e::def_to_str,
          tcx: self.ccx.tcx,
          reachable: encoder::reachable(self, _),
          abbrevs: tyencode::ac_use_abbrevs(self.type_abbrevs)}
    }
}

impl helpers for ebml::writer {
    fn emit_ty(ecx: @e::encode_ctxt, ty: ty::t) {
        e::write_type(ecx, self, ty)
    }

    fn emit_tys(ecx: @e::encode_ctxt, tys: [ty::t]) {
        self.emit_from_vec(tys) {|ty|
            e::write_type(ecx, self, ty)
        }
    }

    fn emit_bounds(ecx: @e::encode_ctxt, bs: ty::param_bounds) {
        tyencode::enc_bounds(self.writer, ecx.ty_str_ctxt(), bs)
    }

    fn emit_tpbt(ecx: @e::encode_ctxt, tpbt: ty::ty_param_bounds_and_ty) {
        self.emit_rec {||
            self.emit_rec_field("bounds", 0u) {||
                self.emit_from_vec(*tpbt.bounds) {|bs|
                    self.emit_bounds(ecx, bs)
                }
            }
            self.emit_rec_field("rp", 1u) {||
                ast::serialize_region_param(self, tpbt.rp)
            }
            self.emit_rec_field("ty", 2u) {||
                self.emit_ty(ecx, tpbt.ty);
            }
        }
    }
}

impl writer for ebml::writer {
    fn tag(tag_id: c::astencode_tag, f: fn()) {
        self.wr_tag(tag_id as uint) {|| f() }
    }

    fn id(id: ast::node_id) {
        self.wr_tagged_u64(c::tag_table_id as uint, id as u64)
    }
}

fn encode_side_tables_for_ii(ecx: @e::encode_ctxt,
                             ebml_w: ebml::writer,
                             ii: ast::inlined_item) {
    ebml_w.wr_tag(c::tag_table as uint) {||
        visit_ids_for_inlined_item(ii, fn@(id: ast::node_id) {
            // Note: this will cause a copy of ebml_w, which is bad as
            // it has mut fields.  But I believe it's harmless since
            // we generate balanced EBML.
            encode_side_tables_for_id(ecx, ebml_w, id)
        });
    }
}

fn encode_side_tables_for_id(ecx: @e::encode_ctxt,
                             ebml_w: ebml::writer,
                             id: ast::node_id) {
    let ccx = ecx.ccx;
    let tcx = ccx.tcx;

    #debug["Encoding side tables for id %d", id];

    option::iter(tcx.def_map.find(id)) {|def|
        ebml_w.tag(c::tag_table_def) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
                ast::serialize_def(ebml_w, def)
            }
        }
    }
    option::iter((*tcx.node_types).find(id as uint)) {|ty|
        ebml_w.tag(c::tag_table_node_type) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
                e::write_type(ecx, ebml_w, ty)
            }
        }
    }

    option::iter(tcx.node_type_substs.find(id)) {|tys|
        ebml_w.tag(c::tag_table_node_type_subst) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
                ebml_w.emit_tys(ecx, tys)
            }
        }
    }

    option::iter(tcx.freevars.find(id)) {|fv|
        ebml_w.tag(c::tag_table_freevars) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
                ebml_w.emit_from_vec(*fv) {|fv_entry|
                    encode_freevar_entry(ebml_w, *fv_entry)
                }
            }
        }
    }

    let lid = {crate: ast::local_crate, node: id};
    option::iter(tcx.tcache.find(lid)) {|tpbt|
        ebml_w.tag(c::tag_table_tcache) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
                ebml_w.emit_tpbt(ecx, tpbt);
            }
        }
    }

    option::iter(tcx.ty_param_bounds.find(id)) {|pbs|
        ebml_w.tag(c::tag_table_param_bounds) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
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

    option::iter(ccx.maps.mutbl_map.find(id)) {|_m|
        ebml_w.tag(c::tag_table_mutbl) {||
            ebml_w.id(id);
        }
    }

    option::iter(ccx.maps.copy_map.find(id)) {|_m|
        ebml_w.tag(c::tag_table_copy) {||
            ebml_w.id(id);
        }
    }

    option::iter(ccx.maps.spill_map.find(id)) {|_m|
        ebml_w.tag(c::tag_table_spill) {||
            ebml_w.id(id);
        }
    }

    option::iter(ccx.maps.last_uses.find(id)) {|m|
        ebml_w.tag(c::tag_table_last_use) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
                last_use::serialize_is_last_use(ebml_w, m)
            }
        }
    }

    // impl_map is not used except when emitting metadata,
    // don't need to keep it.

    option::iter(ccx.maps.method_map.find(id)) {|mo|
        ebml_w.tag(c::tag_table_method_map) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
                serialize_method_origin(ebml_w, mo)
            }
        }
    }

    option::iter(ccx.maps.vtable_map.find(id)) {|dr|
        ebml_w.tag(c::tag_table_vtable_map) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
                encode_vtable_res(ecx, ebml_w, dr);
            }
        }
    }

    option::iter(tcx.borrowings.find(id)) {|_i|
        ebml_w.tag(c::tag_table_borrowings) {||
            ebml_w.id(id);
        }
    }
}

impl decoder for ebml::doc {
    fn as_int() -> int { ebml::doc_as_u64(self) as int }
    fn [](tag: c::astencode_tag) -> ebml::doc {
        ebml::get_doc(self, tag as uint)
    }
    fn opt_child(tag: c::astencode_tag) -> option<ebml::doc> {
        ebml::maybe_get_doc(self, tag as uint)
    }
}

impl decoder for ebml::ebml_deserializer {
    fn read_ty(xcx: extended_decode_ctxt) -> ty::t {
        // Note: regions types embed local node ids.  In principle, we
        // should translate these node ids into the new decode
        // context.  However, we do not bother, because region types
        // are not used during trans.

        tydecode::parse_ty_data(
            self.parent.data, xcx.dcx.cdata.cnum, self.pos, xcx.dcx.tcx,
            xcx.tr_def_id(_))
    }

    fn read_tys(xcx: extended_decode_ctxt) -> [ty::t] {
        self.read_to_vec {|| self.read_ty(xcx) }
    }

    fn read_bounds(xcx: extended_decode_ctxt) -> @[ty::param_bound] {
        tydecode::parse_bounds_data(
            self.parent.data, self.pos, xcx.dcx.cdata.cnum, xcx.dcx.tcx,
            xcx.tr_def_id(_))
    }

    fn read_ty_param_bounds_and_ty(xcx: extended_decode_ctxt)
        -> ty::ty_param_bounds_and_ty {
        self.read_rec {||
            {
                bounds: self.read_rec_field("bounds", 0u) {||
                    @self.read_to_vec {|| self.read_bounds(xcx) }
                },
                rp: self.read_rec_field("rp", 1u) {||
                    ast::deserialize_region_param(self)
                },
                ty: self.read_rec_field("ty", 2u) {||
                    self.read_ty(xcx)
                }
            }
        }
    }
}

fn decode_side_tables(xcx: extended_decode_ctxt,
                      ast_doc: ebml::doc) {
    let dcx = xcx.dcx;
    let tbl_doc = ast_doc[c::tag_table];
    ebml::docs(tbl_doc) {|tag, entry_doc|
        let id0 = entry_doc[c::tag_table_id].as_int();
        let id = xcx.tr_id(id0);

        #debug[">> Side table document with tag 0x%x \
                found for id %d (orig %d)",
               tag, id, id0];

        if tag == (c::tag_table_mutbl as uint) {
            dcx.maps.mutbl_map.insert(id, ());
        } else if tag == (c::tag_table_copy as uint) {
            dcx.maps.copy_map.insert(id, ());
        } else if tag == (c::tag_table_spill as uint) {
            dcx.maps.spill_map.insert(id, ());
        } else if tag == (c::tag_table_borrowings as uint) {
            dcx.tcx.borrowings.insert(id, ());
        } else {
            let val_doc = entry_doc[c::tag_table_val];
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
                let fv_info = @val_dsr.read_to_vec {||
                    @val_dsr.read_freevar_entry(xcx)
                };
                dcx.tcx.freevars.insert(id, fv_info);
            } else if tag == (c::tag_table_tcache as uint) {
                let tpbt = val_dsr.read_ty_param_bounds_and_ty(xcx);
                let lid = {crate: ast::local_crate, node: id};
                dcx.tcx.tcache.insert(lid, tpbt);
            } else if tag == (c::tag_table_param_bounds as uint) {
                let bounds = val_dsr.read_bounds(xcx);
                dcx.tcx.ty_param_bounds.insert(id, bounds);
            } else if tag == (c::tag_table_last_use as uint) {
                dcx.maps.last_uses.insert(id, val_dsr.read_is_last_use(xcx));
            } else if tag == (c::tag_table_method_map as uint) {
                dcx.maps.method_map.insert(id,
                                           val_dsr.read_method_origin(xcx));
            } else if tag == (c::tag_table_vtable_map as uint) {
                dcx.maps.vtable_map.insert(id,
                                         val_dsr.read_vtable_res(xcx));
            } else {
                xcx.dcx.tcx.sess.bug(
                    #fmt["unknown tag found in side tables: %x", tag]);
            }
        }

        #debug[">< Side table doc loaded"];
    }
}

// ______________________________________________________________________
// Testing of astencode_gen

#[cfg(test)]
fn encode_item_ast(ebml_w: ebml::writer, item: @ast::item) {
    ebml_w.wr_tag(c::tag_tree as uint) {||
        ast::serialize_item(ebml_w, *item);
    }
}

#[cfg(test)]
fn decode_item_ast(par_doc: ebml::doc) -> @ast::item {
    let chi_doc = par_doc[c::tag_tree];
    let d = ebml::ebml_deserializer(chi_doc);
    @ast::deserialize_item(d)
}

#[cfg(test)]
fn new_parse_sess() -> parse::parse_sess {
    let cm = codemap::new_codemap();
    let handler = diagnostic::mk_handler(option::none);
    let sess = @{
        cm: cm,
        mut next_id: 1,
        span_diagnostic: diagnostic::mk_span_handler(handler, cm),
        mut chpos: 0u,
        mut byte_pos: 0u
    };
    ret sess;
}

#[cfg(test)]
iface fake_ext_ctxt {
    fn cfg() -> ast::crate_cfg;
    fn parse_sess() -> parse::parse_sess;
}

#[cfg(test)]
type fake_session = ();

#[cfg(test)]
impl of fake_ext_ctxt for fake_session {
    fn cfg() -> ast::crate_cfg { [] }
    fn parse_sess() -> parse::parse_sess { new_parse_sess() }
}

#[cfg(test)]
fn mk_ctxt() -> fake_ext_ctxt {
    () as fake_ext_ctxt
}

#[cfg(test)]
fn roundtrip(in_item: @ast::item) {
    #debug["in_item = %s", pprust::item_to_str(in_item)];
    let mbuf = io::mem_buffer();
    let ebml_w = ebml::writer(io::mem_buffer_writer(mbuf));
    encode_item_ast(ebml_w, in_item);
    let ebml_doc = ebml::doc(@io::mem_buffer_buf(mbuf));
    let out_item = decode_item_ast(ebml_doc);
    #debug["out_item = %s", pprust::item_to_str(out_item)];

    let exp_str =
        io::with_str_writer {|w| ast::serialize_item(w, *in_item) };
    let out_str =
        io::with_str_writer {|w| ast::serialize_item(w, *out_item) };

    #debug["expected string: %s", exp_str];
    #debug["actual string  : %s", out_str];

    assert exp_str == out_str;
}

#[test]
fn test_basic() {
    let ext_cx = mk_ctxt();
    roundtrip(#ast(item){
        fn foo() {}
    });
}

#[test]
fn test_smalltalk() {
    let ext_cx = mk_ctxt();
    roundtrip(#ast(item){
        fn foo() -> int { 3 + 4 } // first smalltalk program ever executed.
    });
}

#[test]
fn test_more() {
    let ext_cx = mk_ctxt();
    roundtrip(#ast(item){
        fn foo(x: uint, y: uint) -> uint {
            let z = x + y;
            ret z;
        }
    });
}

#[test]
fn test_simplification() {
    let ext_cx = mk_ctxt();
    let item_in = ast::ii_item(#ast(item) {
        fn new_int_alist<B: copy>() -> alist<int, B> {
            fn eq_int(&&a: int, &&b: int) -> bool { a == b }
            ret {eq_fn: eq_int, mut data: []};
        }
    });
    let item_out = simplify_ast(item_in);
    let item_exp = ast::ii_item(#ast(item) {
        fn new_int_alist<B: copy>() -> alist<int, B> {
            ret {eq_fn: eq_int, mut data: []};
        }
    });
    alt (item_out, item_exp) {
      (ast::ii_item(item_out), ast::ii_item(item_exp)) {
        assert pprust::item_to_str(item_out) == pprust::item_to_str(item_exp);
      }
      _ { fail; }
    }
}
