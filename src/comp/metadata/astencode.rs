import syntax::ast;
import syntax::fold;
import syntax::visit;
import syntax::ast_util;
import syntax::codemap::span;
import std::map::map;
import std::smallintmap::map;
import std::ebml;
import std::ebml::writer;
import std::serialization;
import std::serialization::serializer;
import std::serialization::deserializer;
import std::serialization::serializer_helpers;
import std::serialization::deserializer_helpers;
import middle::trans::common::maps;
import middle::ty;
import middle::typeck;
import middle::typeck::method_origin;
import middle::typeck::dict_res;
import middle::typeck::dict_origin;
import middle::ast_map;
import driver::session;
import driver::session::session;
import middle::freevars::freevar_entry;
import c = common;
import e = encoder;

// used in testing:
import std::io;
import driver::diagnostic;
import syntax::codemap;
import syntax::parse::parser;
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
// Enumerating the IDs which appear in an AST

fn encode_inlined_item(ecx: @e::encode_ctxt,
                       ebml_w: ebml::writer,
                       path: ast_map::path,
                       item: @ast::item) {
    #debug["> Encoding inlined item: %s::%s (%u)",
           ast_map::path_to_str(path),
           item.ident,
           ebml_w.writer.tell()];
    let id_range = compute_id_range(item);
    ebml_w.wr_tag(c::tag_ast as uint) {||
        encode_id_range(ebml_w, id_range);
        encode_ast(ebml_w, item);
        encode_side_tables_for_item(ecx, ebml_w, item);
    }
    #debug["< Encoded inlined item: %s (%u)",
           ast_map::path_to_str(path),
           ebml_w.writer.tell()];
}

fn decode_inlined_item(cdata: cstore::crate_metadata,
                       tcx: ty::ctxt,
                       maps: maps,
                       path: ast_map::path,
                       par_doc: ebml::doc) -> option<@ast::item> {
    let dcx = @{cdata: cdata, tcx: tcx, maps: maps};
    alt par_doc.opt_child(c::tag_ast) {
      none { none }
      some(ast_doc) {
        #debug["> Decoding inlined item: %s", ast_map::path_to_str(path)];
        let from_id_range = decode_id_range(ast_doc);
        let to_id_range = reserve_id_range(dcx.tcx.sess, from_id_range);
        let xcx = @{dcx: dcx,
                    from_id_range: from_id_range,
                    to_id_range: to_id_range};
        let raw_item = decode_ast(ast_doc);
        let item = renumber_ast(xcx, raw_item);
        #debug[">> Item named: %s", item.ident];
        ast_map::map_decoded_item(dcx.tcx.items, path, item);
        decode_side_tables(xcx, ast_doc);
        #debug["< Decoded inlined item: %s", ast_map::path_to_str(path)];
        some(item)
      }
    }
}

// ______________________________________________________________________
// Enumerating the IDs which appear in an AST

type id_range = {min: ast::node_id, max: ast::node_id};

fn empty(range: id_range) -> bool {
    range.min >= range.max
}

fn visit_ids(item: @ast::item, vfn: fn@(ast::node_id)) {
    let visitor = visit::mk_simple_visitor(@{
        visit_mod: fn@(_m: ast::_mod, _sp: span, id: ast::node_id) {
            vfn(id)
        },

        visit_view_item: fn@(vi: @ast::view_item) {
            alt vi.node {
              ast::view_item_use(_, _, id) { vfn(id) }
              ast::view_item_import(vps) | ast::view_item_export(vps) {
                vec::iter(vps) {|vp|
                    alt vp.node {
                      ast::view_path_simple(_, _, id) { vfn(id) }
                      ast::view_path_glob(_, id) { vfn(id) }
                      ast::view_path_list(_, _, id) { vfn(id) }
                    }
                }
              }
            }
        },

        visit_native_item: fn@(ni: @ast::native_item) {
            vfn(ni.id)
        },

        visit_item: fn@(i: @ast::item) {
            vfn(i.id)
        },

        visit_local: fn@(l: @ast::local) {
            vfn(l.node.id);
        },

        visit_block: fn@(b: ast::blk) {
            vfn(b.node.id);
        },

        visit_stmt: fn@(s: @ast::stmt) {
            vfn(ast_util::stmt_id(*s));
        },

        visit_arm: fn@(_a: ast::arm) { },

        visit_pat: fn@(p: @ast::pat) {
            vfn(p.id)
        },

        visit_decl: fn@(_d: @ast::decl) {
        },

        visit_expr: fn@(e: @ast::expr) {
            vfn(e.id);
            alt e.node {
              ast::expr_unary(_, _) | ast::expr_binary(_, _, _) {
                vfn(ast_util::op_expr_callee_id(e));
              }
              _ { /* fallthrough */ }
            }
        },

        visit_ty: fn@(t: @ast::ty) {
            alt t.node {
              ast::ty_path(_, id) {
                vfn(id)
              }
              _ { /* fall through */ }
            }
        },

        visit_ty_params: fn@(ps: [ast::ty_param]) {
            vec::iter(ps) {|p| vfn(p.id) }
        },

        visit_constr: fn@(_p: @ast::path, _sp: span, id: ast::node_id) {
            vfn(id);
        },

        visit_fn: fn@(fk: visit::fn_kind, d: ast::fn_decl,
                      _b: ast::blk, _sp: span, id: ast::node_id) {
            vfn(id);

            alt fk {
              visit::fk_item_fn(_, tps) |
              visit::fk_method(_, tps) |
              visit::fk_res(_, tps) {
                vec::iter(tps) {|tp| vfn(tp.id)}
              }
              visit::fk_anon(_) |
              visit::fk_fn_block {
              }
            }

            vec::iter(d.inputs) {|arg|
                vfn(arg.id)
            }
        },

        visit_class_item: fn@(_s: span, _p: ast::privacy,
                              c: ast::class_member) {
            alt c {
              ast::instance_var(_, _, _, id) {
                vfn(id)
              }
              ast::class_method(_) {
              }
            }
        }
    });

    visitor.visit_item(item, (), visitor);
}

fn compute_id_range(item: @ast::item) -> id_range {
    let min = @mutable int::max_value;
    let max = @mutable int::min_value;
    visit_ids(item) {|id|
        *min = int::min(*min, id);
        *max = int::max(*max, id + 1);
    }
    ret {min:*min, max:*max};
}

fn encode_id_range(ebml_w: ebml::writer, id_range: id_range) {
    ebml_w.wr_tag(c::tag_id_range as uint) {||
        ebml_w.emit_tup(2u) {||
            ebml_w.emit_tup_elt(0u) {|| ebml_w.emit_int(id_range.min) }
            ebml_w.emit_tup_elt(1u) {|| ebml_w.emit_int(id_range.max) }
        }
    }
}

fn decode_id_range(par_doc: ebml::doc) -> id_range {
    let range_doc = par_doc[c::tag_id_range];
    let dsr = serialization::mk_ebml_deserializer(range_doc);
    dsr.read_tup(2u) {||
        {min: dsr.read_tup_elt(0u) {|| dsr.read_int() },
         max: dsr.read_tup_elt(1u) {|| dsr.read_int() }}
    }
}

fn reserve_id_range(sess: session::session,
                    from_id_range: id_range) -> id_range {
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

impl serializer_helpers<S: serialization::serializer> for S {
    fn emit_def_id(did: ast::def_id) {
        astencode_gen::serialize_syntax_ast_def_id(self, did)
    }
}

impl deserializer_helpers<D: serialization::deserializer> for D {
    fn read_def_id(xcx: extended_decode_ctxt) -> ast::def_id {
        let did = astencode_gen::deserialize_syntax_ast_def_id(self);
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

fn encode_ast(ebml_w: ebml::writer, item: @ast::item) {
    ebml_w.wr_tag(c::tag_tree as uint) {||
        astencode_gen::serialize_syntax_ast_item(ebml_w, *item);
    }
}

fn decode_ast(par_doc: ebml::doc) -> @ast::item {
    let chi_doc = par_doc[c::tag_tree];
    let d = serialization::mk_ebml_deserializer(chi_doc);
    @astencode_gen::deserialize_syntax_ast_item(d)
}

fn renumber_ast(xcx: extended_decode_ctxt, item: @ast::item) -> @ast::item {
    let fld = fold::make_fold({
        new_id: xcx.tr_id(_),
        new_span: xcx.tr_span(_)
        with *fold::default_ast_fold()
    });
    fld.fold_item(item)
}

// ______________________________________________________________________
// Encoding and decoding of ast::def

fn encode_def(ebml_w: ebml::writer, def: ast::def) {
    astencode_gen::serialize_syntax_ast_def(ebml_w, def)
}

fn decode_def(xcx: extended_decode_ctxt, doc: ebml::doc) -> ast::def {
    let dsr = serialization::mk_ebml_deserializer(doc);
    let def = astencode_gen::deserialize_syntax_ast_def(dsr);
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
          ast::def_local(nid) { ast::def_local(xcx.tr_id(nid)) }
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
          ast::def_class_field(did0, did1) {
            ast::def_class_field(did0.tr(xcx), did1.tr(xcx))
          }
          ast::def_class_method(did0, did1) {
            ast::def_class_method(did0.tr(xcx), did1.tr(xcx))
          }
        }
    }
}

// ______________________________________________________________________
// Encoding and decoding of freevar information

fn encode_freevar_entry(ebml_w: ebml::writer, fv: freevar_entry) {
    astencode_gen::serialize_middle_freevars_freevar_entry(ebml_w, fv)
}

impl helper for serialization::ebml_deserializer {
    fn read_freevar_entry(xcx: extended_decode_ctxt) -> freevar_entry {
        let fv =
            astencode_gen::deserialize_middle_freevars_freevar_entry(self);
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
    astencode_gen::serialize_middle_typeck_method_origin(ebml_w, mo)
}

impl helper for serialization::ebml_deserializer {
    fn read_method_origin(xcx: extended_decode_ctxt) -> method_origin {
        let fv = astencode_gen::deserialize_middle_typeck_method_origin(self);
        fv.tr(xcx)
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

// ______________________________________________________________________
// Encoding and decoding dict_res

fn encode_dict_res(ecx: @e::encode_ctxt,
                   ebml_w: ebml::writer,
                   dr: typeck::dict_res) {
    // can't autogenerate this code because automatic serialization of
    // ty::t doesn't work, and there is no way (atm) to have
    // hand-written serialization routines combine with auto-generated
    // ones.  perhaps we should fix this.
    ebml_w.emit_from_vec(*dr) {|dict_origin|
        encode_dict_origin(ecx, ebml_w, dict_origin)
    }
}

fn encode_dict_origin(ecx: @e::encode_ctxt,
                      ebml_w: ebml::writer,
                      dict_origin: typeck::dict_origin) {
    ebml_w.emit_enum("dict_origin") {||
        alt dict_origin {
          typeck::dict_static(def_id, tys, dict_res) {
            ebml_w.emit_enum_variant("dict_static", 0u, 3u) {||
                ebml_w.emit_enum_variant_arg(0u) {||
                    ebml_w.emit_def_id(def_id)
                }
                ebml_w.emit_enum_variant_arg(1u) {||
                    ebml_w.emit_tys(ecx, tys);
                }
                ebml_w.emit_enum_variant_arg(2u) {||
                    encode_dict_res(ecx, ebml_w, dict_res);
                }
            }
          }
          typeck::dict_param(pn, bn) {
            ebml_w.emit_enum_variant("dict_param", 1u, 2u) {||
                ebml_w.emit_enum_variant_arg(0u) {||
                    ebml_w.emit_uint(pn);
                }
                ebml_w.emit_enum_variant_arg(1u) {||
                    ebml_w.emit_uint(bn);
                }
            }
          }
          typeck::dict_iface(def_id) {
            ebml_w.emit_enum_variant("dict_iface", 1u, 3u) {||
                ebml_w.emit_enum_variant_arg(0u) {||
                    ebml_w.emit_def_id(def_id)
                }
            }
          }
        }
    }

}

impl helpers for serialization::ebml_deserializer {
    fn read_dict_res(xcx: extended_decode_ctxt) -> typeck::dict_res {
        @self.read_to_vec {|| self.read_dict_origin(xcx) }
    }

    fn read_dict_origin(xcx: extended_decode_ctxt) -> typeck::dict_origin {
        self.read_enum("dict_origin") {||
            self.read_enum_variant {|i|
                alt check i {
                  0u {
                    typeck::dict_static(
                        self.read_enum_variant_arg(0u) {||
                            self.read_def_id(xcx)
                        },
                        self.read_enum_variant_arg(1u) {||
                            self.read_tys(xcx)
                        },
                        self.read_enum_variant_arg(2u) {||
                            self.read_dict_res(xcx)
                        }
                    )
                  }
                  1u {
                    typeck::dict_param(
                        self.read_enum_variant_arg(0u) {||
                            self.read_uint()
                        },
                        self.read_enum_variant_arg(1u) {||
                            self.read_uint()
                        }
                    )
                  }
                  2u {
                    typeck::dict_iface(
                        self.read_enum_variant_arg(0u) {||
                            self.read_def_id(xcx)
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
            self.emit_rec_field("ty", 0u) {||
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

fn encode_side_tables_for_item(ecx: @e::encode_ctxt,
                               ebml_w: ebml::writer,
                               item: @ast::item) {
    ebml_w.wr_tag(c::tag_table as uint) {||
        visit_ids(item, fn@(id: ast::node_id) {
            // Note: this will cause a copy of ebml_w, which is bad as
            // it has mutable fields.  But I believe it's harmless since
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

    option::may(tcx.def_map.find(id)) {|def|
        ebml_w.tag(c::tag_table_def) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
                astencode_gen::serialize_syntax_ast_def(ebml_w, def)
            }
        }
    }
    option::may((*tcx.node_types).find(id as uint)) {|ty|
        ebml_w.tag(c::tag_table_node_type) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
                e::write_type(ecx, ebml_w, ty)
            }
        }
    }

    option::may(tcx.node_type_substs.find(id)) {|tys|
        ebml_w.tag(c::tag_table_node_type_subst) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
                ebml_w.emit_tys(ecx, tys)
            }
        }
    }

    option::may(tcx.freevars.find(id)) {|fv|
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
    option::may(tcx.tcache.find(lid)) {|tpbt|
        ebml_w.tag(c::tag_table_tcache) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
                ebml_w.emit_tpbt(ecx, tpbt);
            }
        }
    }

    option::may(tcx.ty_param_bounds.find(id)) {|pbs|
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
    //option::may(tcx.inferred_modes.find(id)) {|m|
    //    ebml_w.tag(c::tag_table_inferred_modes) {||
    //        ebml_w.id(id);
    //        ebml_w.tag(c::tag_table_val) {||
    //            tyencode::enc_mode(ebml_w.writer, ty_str_ctxt(), m);
    //        }
    //    }
    //}

    option::may(ccx.maps.mutbl_map.find(id)) {|_m|
        ebml_w.tag(c::tag_table_mutbl) {||
            ebml_w.id(id);
        }
    }

    option::may(ccx.maps.copy_map.find(id)) {|_m|
        ebml_w.tag(c::tag_table_copy) {||
            ebml_w.id(id);
        }
    }

    option::may(ccx.maps.last_uses.find(id)) {|_m|
        ebml_w.tag(c::tag_table_last_use) {||
            ebml_w.id(id);
        }
    }

    // impl_map is not used except when emitting metadata,
    // don't need to keep it.

    option::may(ccx.maps.method_map.find(id)) {|mo|
        ebml_w.tag(c::tag_table_method_map) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
                astencode_gen::
                    serialize_middle_typeck_method_origin(ebml_w, mo)
            }
        }
    }

    option::may(ccx.maps.dict_map.find(id)) {|dr|
        ebml_w.tag(c::tag_table_dict_map) {||
            ebml_w.id(id);
            ebml_w.tag(c::tag_table_val) {||
                encode_dict_res(ecx, ebml_w, dr);
            }
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

impl decoder for serialization::ebml_deserializer {
    fn read_ty(xcx: extended_decode_ctxt) -> ty::t {
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
                ty: self.read_rec_field("ty", 1u) {||
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

        #debug[">> Side table document with tag 0x%x found for id %d (orig %d)",
               tag, id, id0];

        if tag == (c::tag_table_mutbl as uint) {
            dcx.maps.mutbl_map.insert(id, ());
        } else if tag == (c::tag_table_copy as uint) {
            dcx.maps.copy_map.insert(id, ());
        } else if tag == (c::tag_table_last_use as uint) {
            dcx.maps.last_uses.insert(id, ());
        } else {
            let val_doc = entry_doc[c::tag_table_val];
            let val_dsr = serialization::mk_ebml_deserializer(val_doc);
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
            } else if tag == (c::tag_table_method_map as uint) {
                dcx.maps.method_map.insert(id,
                                           val_dsr.read_method_origin(xcx));
            } else if tag == (c::tag_table_dict_map as uint) {
                dcx.maps.dict_map.insert(id,
                                         val_dsr.read_dict_res(xcx));
            } else {
                xcx.dcx.tcx.sess.bug(
                    #fmt["Unknown tag found in side tables: %x", tag]);
            }
        }

        #debug[">< Side table doc loaded"];
    }
}

// ______________________________________________________________________
// Testing

#[cfg(test)]
fn new_parse_sess() -> parser::parse_sess {
    let cm = codemap::new_codemap();
    let handler = diagnostic::mk_handler(option::none);
    let sess = @{
        cm: cm,
        mutable next_id: 1,
        span_diagnostic: diagnostic::mk_span_handler(handler, cm),
        mutable chpos: 0u,
        mutable byte_pos: 0u
    };
    ret sess;
}

#[cfg(test)]
iface fake_ext_ctxt {
    fn session() -> fake_session;
}

#[cfg(test)]
type fake_options = {cfg: ast::crate_cfg};

#[cfg(test)]
type fake_session = {opts: @fake_options,
                     parse_sess: parser::parse_sess};

#[cfg(test)]
impl of fake_ext_ctxt for fake_session {
    fn session() -> fake_session {self}
}

#[cfg(test)]
fn mk_ctxt() -> fake_ext_ctxt {
    let opts : fake_options = {cfg: []};
    {opts: @opts, parse_sess: new_parse_sess()} as fake_ext_ctxt
}

#[cfg(test)]
fn roundtrip(in_item: @ast::item) {
    #debug["in_item = %s", pprust::item_to_str(in_item)];
    let mbuf = io::mk_mem_buffer();
    let ebml_w = ebml::mk_writer(io::mem_buffer_writer(mbuf));
    encode_ast(ebml_w, in_item);
    let ebml_doc = ebml::new_doc(@io::mem_buffer_buf(mbuf));
    let out_item = decode_ast(ebml_doc);
    #debug["out_item = %s", pprust::item_to_str(out_item)];
    assert in_item == out_item;
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