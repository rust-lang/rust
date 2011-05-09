import std.Str;
import std.UInt;
import std.Vec;
import std.Map.hashmap;
import std.EBML;
import std.IO;
import std.Option;
import std.Option.some;
import std.Option.none;

import front.ast;
import middle.fold;
import middle.trans;
import middle.ty;
import back.x86;
import util.common;

import lib.llvm.llvm;
import lib.llvm.llvm.ValueRef;
import lib.llvm.False;

const uint tag_paths = 0x01u;
const uint tag_items = 0x02u;

const uint tag_paths_data = 0x03u;
const uint tag_paths_data_name = 0x04u;
const uint tag_paths_data_item = 0x05u;
const uint tag_paths_data_mod = 0x06u;

const uint tag_def_id = 0x07u;

const uint tag_items_data = 0x08u;
const uint tag_items_data_item = 0x09u;
const uint tag_items_data_item_kind = 0x0au;
const uint tag_items_data_item_ty_param_count = 0x0bu;
const uint tag_items_data_item_type = 0x0cu;
const uint tag_items_data_item_symbol = 0x0du;
const uint tag_items_data_item_variant = 0x0eu;
const uint tag_items_data_item_tag_id = 0x0fu;
const uint tag_items_data_item_obj_type_id = 0x10u;

const uint tag_index = 0x11u;
const uint tag_index_buckets = 0x12u;
const uint tag_index_buckets_bucket = 0x13u;
const uint tag_index_buckets_bucket_elt = 0x14u;
const uint tag_index_table = 0x15u;

// Type encoding

// Compact string representation for ty.t values. API ty_str & parse_from_str.
// Extra parameters are for converting to/from def_ids in the string rep.
// Whatever format you choose should not contain pipe characters.

type ty_abbrev = rec(uint pos, uint len, str s);

tag abbrev_ctxt {
    ac_no_abbrevs;
    ac_use_abbrevs(hashmap[ty.t, ty_abbrev]);
}

mod Encode {

    type ctxt = rec(
        fn(&ast.def_id) -> str ds,          // Def -> str Callback.
        ty.ctxt tcx,                        // The type context.
        abbrev_ctxt abbrevs
    );

    fn cx_uses_abbrevs(&@ctxt cx) -> bool {
        alt (cx.abbrevs) {
            case (ac_no_abbrevs)     { ret false; }
            case (ac_use_abbrevs(_)) { ret true; }
        }
    }

    fn ty_str(&@ctxt cx, &ty.t t) -> str {
        assert (!cx_uses_abbrevs(cx));
        auto sw = IO.string_writer();
        enc_ty(sw.get_writer(), cx, t);
        ret sw.get_str();
    }

    fn enc_ty(&IO.writer w, &@ctxt cx, &ty.t t) {
        alt (cx.abbrevs) {
            case (ac_no_abbrevs) { enc_sty(w, cx, ty.struct(cx.tcx, t)); }
            case (ac_use_abbrevs(?abbrevs)) {
                alt (abbrevs.find(t)) {
                    case (some[ty_abbrev](?a)) {
                        w.write_str(a.s);
                        ret;
                    }
                    case (none[ty_abbrev]) {
                        auto pos = w.get_buf_writer().tell();
                        auto ss = enc_sty(w, cx, ty.struct(cx.tcx, t));
                        auto end = w.get_buf_writer().tell();
                        auto len = end-pos;
                        fn estimate_sz(uint u) -> uint {
                            auto n = u;
                            auto len = 0u;
                            while (n != 0u) {
                                len += 1u;
                                n = n >> 4u;
                            }
                            ret len;
                        }
                        auto abbrev_len =
                            3u + estimate_sz(pos) + estimate_sz(len);

                        if (abbrev_len < len) {
                            // I.e. it's actually an abbreviation.
                            auto s = ("#"
                                      + UInt.to_str(pos, 16u) + ":"
                                      + UInt.to_str(len, 16u) + "#");
                            auto a = rec(pos=pos, len=len, s=s);
                            abbrevs.insert(t, a);
                        }
                        ret;
                    }
                }
            }
        }
    }

    fn enc_mt(&IO.writer w, &@ctxt cx, &ty.mt mt) {
        alt (mt.mut) {
            case (ast.imm)       { }
            case (ast.mut)       { w.write_char('m'); }
            case (ast.maybe_mut) { w.write_char('?'); }
        }
        enc_ty(w, cx, mt.ty);
    }

    fn enc_sty(&IO.writer w, &@ctxt cx, &ty.sty st) {
        alt (st) {
            case (ty.ty_nil) { w.write_char('n'); }
            case (ty.ty_bool) { w.write_char('b'); }
            case (ty.ty_int) { w.write_char('i'); }
            case (ty.ty_uint) { w.write_char('u'); }
            case (ty.ty_float) { w.write_char('l'); }
            case (ty.ty_machine(?mach)) {
                alt (mach) {
                    case (common.ty_u8) { w.write_str("Mb"); }
                    case (common.ty_u16) { w.write_str("Mw"); }
                    case (common.ty_u32) { w.write_str("Ml"); }
                    case (common.ty_u64) { w.write_str("Md"); }
                    case (common.ty_i8) { w.write_str("MB"); }
                    case (common.ty_i16) { w.write_str("MW"); }
                    case (common.ty_i32) { w.write_str("ML"); }
                    case (common.ty_i64) { w.write_str("MD"); }
                    case (common.ty_f32) { w.write_str("Mf"); }
                    case (common.ty_f64) { w.write_str("MF"); }
                }
            }
            case (ty.ty_char) {w.write_char('c');}
            case (ty.ty_str) {w.write_char('s');}
            case (ty.ty_tag(?def,?tys)) { // TODO restore def_id
                w.write_str("t[");
                w.write_str(cx.ds(def));
                w.write_char('|');
                for (ty.t t in tys) {
                    enc_ty(w, cx, t);
                }
                w.write_char(']');
            }
            case (ty.ty_box(?mt)) {w.write_char('@'); enc_mt(w, cx, mt); }
            case (ty.ty_vec(?mt)) {w.write_char('V'); enc_mt(w, cx, mt); }
            case (ty.ty_port(?t)) {w.write_char('P'); enc_ty(w, cx, t); }
            case (ty.ty_chan(?t)) {w.write_char('C'); enc_ty(w, cx, t); }
            case (ty.ty_tup(?mts)) {
                w.write_str("T[");
                for (ty.mt mt in mts) {
                    enc_mt(w, cx, mt);
                }
                w.write_char(']');
            }
            case (ty.ty_rec(?fields)) {
                w.write_str("R[");
                for (ty.field field in fields) {
                    w.write_str(field.ident);
                    w.write_char('=');
                    enc_mt(w, cx, field.mt);
                }
                w.write_char(']');
            }
            case (ty.ty_fn(?proto,?args,?out)) {
                enc_proto(w, proto);
                enc_ty_fn(w, cx, args, out);
            }
            case (ty.ty_native_fn(?abi,?args,?out)) {
                w.write_char('N');
                alt (abi) {
                    case (ast.native_abi_rust) { w.write_char('r'); }
                    case (ast.native_abi_rust_intrinsic) {
                        w.write_char('i');
                    }
                    case (ast.native_abi_cdecl) { w.write_char('c'); }
                    case (ast.native_abi_llvm) { w.write_char('l'); }
                }
                enc_ty_fn(w, cx, args, out);
            }
            case (ty.ty_obj(?methods)) {
                w.write_str("O[");
                for (ty.method m in methods) {
                    enc_proto(w, m.proto);
                    w.write_str(m.ident);
                    enc_ty_fn(w, cx, m.inputs, m.output);
                }
                w.write_char(']');
            }
            case (ty.ty_var(?id)) {
                w.write_char('X');
                w.write_str(common.istr(id));
            }
            case (ty.ty_native) {w.write_char('E');}
            case (ty.ty_param(?id)) {
                w.write_char('p');
                w.write_str(common.uistr(id));
            }
            case (ty.ty_type) {w.write_char('Y');}

            // These two don't appear in crate metadata, but are here because
            // `hash_ty()` uses this function.
            case (ty.ty_bound_param(?id)) {
                w.write_char('o');
                w.write_str(common.uistr(id));
            }
            case (ty.ty_local(?def)) {
                w.write_char('L');
                w.write_str(cx.ds(def));
            }
        }
    }

    fn enc_proto(&IO.writer w, ast.proto proto) {
        alt (proto) {
            case (ast.proto_iter) { w.write_char('W'); }
            case (ast.proto_fn) { w.write_char('F'); }
        }
    }

    fn enc_ty_fn(&IO.writer w, &@ctxt cx, &vec[ty.arg] args, &ty.t out) {
        w.write_char('[');
        for (ty.arg arg in args) {
            if (arg.mode == ty.mo_alias) { w.write_char('&'); }
            enc_ty(w, cx, arg.ty);
        }
        w.write_char(']');
        enc_ty(w, cx, out);
    }

}


// Returns a Plain Old LLVM String.
fn C_postr(&str s) -> ValueRef {
    ret llvm.LLVMConstString(Str.buf(s), Str.byte_len(s), False);
}


// Path table encoding

fn encode_name(&EBML.writer ebml_w, &str name) {
    EBML.start_tag(ebml_w, tag_paths_data_name);
    ebml_w.writer.write(Str.bytes(name));
    EBML.end_tag(ebml_w);
}

fn encode_def_id(&EBML.writer ebml_w, &ast.def_id id) {
    EBML.start_tag(ebml_w, tag_def_id);
    ebml_w.writer.write(Str.bytes(def_to_str(id)));
    EBML.end_tag(ebml_w);
}

fn encode_tag_variant_paths(&EBML.writer ebml_w,
                            &vec[ast.variant] variants,
                            &vec[str] path,
                            &mutable vec[tup(str, uint)] index) {
    for (ast.variant variant in variants) {
        add_to_index(ebml_w, path, index, variant.node.name);
        EBML.start_tag(ebml_w, tag_paths_data_item);
        encode_name(ebml_w, variant.node.name);
        encode_def_id(ebml_w, variant.node.id);
        EBML.end_tag(ebml_w);
    }
}

fn add_to_index(&EBML.writer ebml_w,
                &vec[str] path,
                &mutable vec[tup(str, uint)] index,
                &str name) {
    auto full_path = path + vec(name);
    index += vec(tup(Str.connect(full_path, "."), ebml_w.writer.tell()));
}

fn encode_native_module_item_paths(&EBML.writer ebml_w,
                                   &ast.native_mod nmod,
                                   &vec[str] path,
                                   &mutable vec[tup(str, uint)] index) {
    for (@ast.native_item nitem in nmod.items) {
        alt (nitem.node) {
            case (ast.native_item_ty(?id, ?did)) {
                add_to_index(ebml_w, path, index, id);
                EBML.start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                EBML.end_tag(ebml_w);
            }
            case (ast.native_item_fn(?id, _, _, _, ?did, _)) {
                add_to_index(ebml_w, path, index, id);
                EBML.start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                EBML.end_tag(ebml_w);
            }
        }
    }
}

fn encode_module_item_paths(&EBML.writer ebml_w,
                            &ast._mod module,
                            &vec[str] path,
                            &mutable vec[tup(str, uint)] index) {
    // TODO: only encode exported items
    for (@ast.item it in module.items) {
        alt (it.node) {
            case (ast.item_const(?id, _, ?tps, ?did, ?ann)) {
                add_to_index(ebml_w, path, index, id);
                EBML.start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                EBML.end_tag(ebml_w);
            }
            case (ast.item_fn(?id, _, ?tps, ?did, ?ann)) {
                add_to_index(ebml_w, path, index, id);
                EBML.start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                EBML.end_tag(ebml_w);
            }
            case (ast.item_mod(?id, ?_mod, ?did)) {
                add_to_index(ebml_w, path, index, id);
                EBML.start_tag(ebml_w, tag_paths_data_mod);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                encode_module_item_paths(ebml_w, _mod, path + vec(id), index);
                EBML.end_tag(ebml_w);
            }
            case (ast.item_native_mod(?id, ?nmod, ?did)) {
                add_to_index(ebml_w, path, index, id);
                EBML.start_tag(ebml_w, tag_paths_data_mod);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                encode_native_module_item_paths(ebml_w, nmod, path + vec(id),
                                                index);
                EBML.end_tag(ebml_w);
            }
            case (ast.item_ty(?id, _, ?tps, ?did, ?ann)) {
                add_to_index(ebml_w, path, index, id);
                EBML.start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                EBML.end_tag(ebml_w);
            }
            case (ast.item_tag(?id, ?variants, ?tps, ?did, _)) {
                add_to_index(ebml_w, path, index, id);
                EBML.start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                EBML.end_tag(ebml_w);

                encode_tag_variant_paths(ebml_w, variants, path, index);
            }
            case (ast.item_obj(?id, _, ?tps, ?odid, ?ann)) {
                add_to_index(ebml_w, path, index, id);
                EBML.start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, odid.ctor);
                encode_obj_type_id(ebml_w, odid.ty);
                EBML.end_tag(ebml_w);
            }
        }
    }
}

fn encode_item_paths(&EBML.writer ebml_w, &@ast.crate crate)
        -> vec[tup(str, uint)] {
    let vec[tup(str, uint)] index = vec();
    let vec[str] path = vec();
    EBML.start_tag(ebml_w, tag_paths);
    encode_module_item_paths(ebml_w, crate.node.module, path, index);
    EBML.end_tag(ebml_w);
    ret index;
}


// Item info table encoding

fn encode_kind(&EBML.writer ebml_w, u8 c) {
    EBML.start_tag(ebml_w, tag_items_data_item_kind);
    ebml_w.writer.write(vec(c));
    EBML.end_tag(ebml_w);
}

fn def_to_str(&ast.def_id did) -> str {
    ret #fmt("%d:%d", did._0, did._1);
}

fn encode_type_param_count(&EBML.writer ebml_w, &vec[ast.ty_param] tps) {
    EBML.start_tag(ebml_w, tag_items_data_item_ty_param_count);
    EBML.write_vint(ebml_w.writer, Vec.len[ast.ty_param](tps));
    EBML.end_tag(ebml_w);
}

fn encode_variant_id(&EBML.writer ebml_w, &ast.def_id vid) {
    EBML.start_tag(ebml_w, tag_items_data_item_variant);
    ebml_w.writer.write(Str.bytes(def_to_str(vid)));
    EBML.end_tag(ebml_w);
}

fn encode_type(&@trans.crate_ctxt cx, &EBML.writer ebml_w, &ty.t typ) {
    EBML.start_tag(ebml_w, tag_items_data_item_type);

    auto f = def_to_str;
    auto ty_str_ctxt = @rec(ds=f, tcx=cx.tcx,
                            abbrevs=ac_use_abbrevs(cx.type_abbrevs));
    Encode.enc_ty(IO.new_writer_(ebml_w.writer), ty_str_ctxt, typ);
    EBML.end_tag(ebml_w);
}

fn encode_symbol(&@trans.crate_ctxt cx, &EBML.writer ebml_w,
                 &ast.def_id did) {
    EBML.start_tag(ebml_w, tag_items_data_item_symbol);
    ebml_w.writer.write(Str.bytes(cx.item_symbols.get(did)));
    EBML.end_tag(ebml_w);
}

fn encode_discriminant(&@trans.crate_ctxt cx, &EBML.writer ebml_w,
                       &ast.def_id did) {
    EBML.start_tag(ebml_w, tag_items_data_item_symbol);
    ebml_w.writer.write(Str.bytes(cx.discrim_symbols.get(did)));
    EBML.end_tag(ebml_w);
}

fn encode_tag_id(&EBML.writer ebml_w, &ast.def_id id) {
    EBML.start_tag(ebml_w, tag_items_data_item_tag_id);
    ebml_w.writer.write(Str.bytes(def_to_str(id)));
    EBML.end_tag(ebml_w);
}

fn encode_obj_type_id(&EBML.writer ebml_w, &ast.def_id id) {
    EBML.start_tag(ebml_w, tag_items_data_item_obj_type_id);
    ebml_w.writer.write(Str.bytes(def_to_str(id)));
    EBML.end_tag(ebml_w);
}


fn encode_tag_variant_info(&@trans.crate_ctxt cx, &EBML.writer ebml_w,
                           &ast.def_id did, &vec[ast.variant] variants,
                           &mutable vec[tup(int, uint)] index,
                           &vec[ast.ty_param] ty_params) {
    for (ast.variant variant in variants) {
        index += vec(tup(variant.node.id._1, ebml_w.writer.tell()));

        EBML.start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, variant.node.id);
        encode_kind(ebml_w, 'v' as u8);
        encode_tag_id(ebml_w, did);
        encode_type(cx, ebml_w, trans.node_ann_type(cx, variant.node.ann));
        if (Vec.len[ast.variant_arg](variant.node.args) > 0u) {
            encode_symbol(cx, ebml_w, variant.node.id);
        }
        encode_discriminant(cx, ebml_w, variant.node.id);
        encode_type_param_count(ebml_w, ty_params);
        EBML.end_tag(ebml_w);
    }
}

fn encode_info_for_item(@trans.crate_ctxt cx, &EBML.writer ebml_w,
                        @ast.item item, &mutable vec[tup(int, uint)] index) {
    alt (item.node) {
        case (ast.item_const(_, _, _, ?did, ?ann)) {
            EBML.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'c' as u8);
            encode_type(cx, ebml_w, trans.node_ann_type(cx, ann));
            encode_symbol(cx, ebml_w, did);
            EBML.end_tag(ebml_w);
        }
        case (ast.item_fn(_, _, ?tps, ?did, ?ann)) {
            EBML.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'f' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, trans.node_ann_type(cx, ann));
            encode_symbol(cx, ebml_w, did);
            EBML.end_tag(ebml_w);
        }
        case (ast.item_mod(_, _, ?did)) {
            EBML.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'm' as u8);
            EBML.end_tag(ebml_w);
        }
        case (ast.item_native_mod(_, _, ?did)) {
            EBML.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'n' as u8);
            EBML.end_tag(ebml_w);
        }
        case (ast.item_ty(?id, _, ?tps, ?did, ?ann)) {
            EBML.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'y' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, trans.node_ann_type(cx, ann));
            EBML.end_tag(ebml_w);
        }
        case (ast.item_tag(?id, ?variants, ?tps, ?did, ?ann)) {
            EBML.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 't' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, trans.node_ann_type(cx, ann));
            for (ast.variant v in variants) {
                encode_variant_id(ebml_w, v.node.id);
            }
            EBML.end_tag(ebml_w);

            encode_tag_variant_info(cx, ebml_w, did, variants, index, tps);
        }
        case (ast.item_obj(?id, _, ?tps, ?odid, ?ann)) {
            EBML.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, odid.ctor);
            encode_kind(ebml_w, 'o' as u8);
            encode_type_param_count(ebml_w, tps);
            auto fn_ty = trans.node_ann_type(cx, ann);
            encode_type(cx, ebml_w, fn_ty);
            encode_symbol(cx, ebml_w, odid.ctor);
            EBML.end_tag(ebml_w);

            EBML.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, odid.ty);
            encode_kind(ebml_w, 'y' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, ty.ty_fn_ret(cx.tcx, fn_ty));
            EBML.end_tag(ebml_w);
        }
    }
}

fn encode_info_for_native_item(&@trans.crate_ctxt cx, &EBML.writer ebml_w,
                               &@ast.native_item nitem) {
    EBML.start_tag(ebml_w, tag_items_data_item);
    alt (nitem.node) {
        case (ast.native_item_ty(_, ?did)) {
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'T' as u8);
            encode_type(cx, ebml_w, ty.mk_native(cx.tcx));
        }
        case (ast.native_item_fn(_, _, _, ?tps, ?did, ?ann)) {
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'F' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, trans.node_ann_type(cx, ann));
            encode_symbol(cx, ebml_w, did);
        }
    }
    EBML.end_tag(ebml_w);
}

fn encode_info_for_items(&@trans.crate_ctxt cx, &EBML.writer ebml_w)
        -> vec[tup(int, uint)] {
    let vec[tup(int, uint)] index = vec();

    EBML.start_tag(ebml_w, tag_items_data);
    for each (@tup(ast.def_id, @ast.item) kvp in cx.items.items()) {
        index += vec(tup(kvp._0._1, ebml_w.writer.tell()));
        encode_info_for_item(cx, ebml_w, kvp._1, index);
    }
    for each (@tup(ast.def_id, @ast.native_item) kvp in
            cx.native_items.items()) {
        index += vec(tup(kvp._0._1, ebml_w.writer.tell()));
        encode_info_for_native_item(cx, ebml_w, kvp._1);
    }
    EBML.end_tag(ebml_w);

    ret index;
}


// Path and definition ID indexing

// djb's cdb hashes.

fn hash_def_num(&int def_num) -> uint {
    ret 177573u ^ (def_num as uint);
}

fn hash_path(&str s) -> uint {
    auto h = 5381u;
    for (u8 ch in Str.bytes(s)) {
        h = ((h << 5u) + h) ^ (ch as uint);
    }
    ret h;
}

fn create_index[T](&vec[tup(T, uint)] index, fn(&T) -> uint hash_fn)
        -> vec[vec[tup(T, uint)]] {
    let vec[vec[tup(T, uint)]] buckets = vec();
    for each (uint i in UInt.range(0u, 256u)) {
        let vec[tup(T, uint)] bucket = vec();
        buckets += vec(bucket);
    }

    for (tup(T, uint) elt in index) {
        auto h = hash_fn(elt._0);
        buckets.(h % 256u) += vec(elt);
    }

    ret buckets;
}

fn encode_index[T](&EBML.writer ebml_w, &vec[vec[tup(T, uint)]] buckets,
                   fn(&IO.writer, &T) write_fn) {
    auto writer = IO.new_writer_(ebml_w.writer);

    EBML.start_tag(ebml_w, tag_index);

    let vec[uint] bucket_locs = vec();
    EBML.start_tag(ebml_w, tag_index_buckets);
    for (vec[tup(T, uint)] bucket in buckets) {
        bucket_locs += vec(ebml_w.writer.tell());

        EBML.start_tag(ebml_w, tag_index_buckets_bucket);
        for (tup(T, uint) elt in bucket) {
            EBML.start_tag(ebml_w, tag_index_buckets_bucket_elt);
            writer.write_be_uint(elt._1, 4u);
            write_fn(writer, elt._0);
            EBML.end_tag(ebml_w);
        }
        EBML.end_tag(ebml_w);
    }
    EBML.end_tag(ebml_w);

    EBML.start_tag(ebml_w, tag_index_table);
    for (uint pos in bucket_locs) {
        writer.write_be_uint(pos, 4u);
    }
    EBML.end_tag(ebml_w);

    EBML.end_tag(ebml_w);
}

fn write_str(&IO.writer writer, &str s) {
    writer.write_str(s);
}

fn write_int(&IO.writer writer, &int n) {
    writer.write_be_uint(n as uint, 4u);
}


fn encode_metadata(&@trans.crate_ctxt cx, &@ast.crate crate)
        -> ValueRef {
    auto string_w = IO.string_writer();
    auto buf_w = string_w.get_writer().get_buf_writer();
    auto ebml_w = EBML.create_writer(buf_w);

    // Encode and index the paths.
    EBML.start_tag(ebml_w, tag_paths);
    auto paths_index = encode_item_paths(ebml_w, crate);
    auto str_writer = write_str;
    auto path_hasher = hash_path;
    auto paths_buckets = create_index[str](paths_index, path_hasher);
    encode_index[str](ebml_w, paths_buckets, str_writer);
    EBML.end_tag(ebml_w);

    // Encode and index the items.
    EBML.start_tag(ebml_w, tag_items);
    auto items_index = encode_info_for_items(cx, ebml_w);
    auto int_writer = write_int;
    auto item_hasher = hash_def_num;
    auto items_buckets = create_index[int](items_index, item_hasher);
    encode_index[int](ebml_w, items_buckets, int_writer);
    EBML.end_tag(ebml_w);

    // Pad this, since something (LLVM, presumably) is cutting off the
    // remaining % 4 bytes.
    buf_w.write(vec(0u8, 0u8, 0u8, 0u8));

    ret C_postr(string_w.get_str());
}

fn write_metadata(&@trans.crate_ctxt cx, &@ast.crate crate) {
    auto llmeta = C_postr("");
    if (cx.sess.get_opts().shared) {
        llmeta = encode_metadata(cx, crate);
    }

    auto llconst = trans.C_struct(vec(llmeta));
    auto llglobal = llvm.LLVMAddGlobal(cx.llmod, trans.val_ty(llconst),
                                       Str.buf("rust_metadata"));
    llvm.LLVMSetInitializer(llglobal, llconst);
    llvm.LLVMSetSection(llglobal, Str.buf(x86.get_meta_sect_name()));
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
