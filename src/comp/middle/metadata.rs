import std::str;
import std::uint;
import std::vec;
import std::map::hashmap;
import std::ebml;
import std::io;
import std::option;
import std::option::some;
import std::option::none;

import front::ast;
import front::ast::*;
import middle::trans;
import middle::ty;
import middle::ty::path_to_str;
import back::x86;
import back::link;
import util::common;
import pretty::ppaux::lit_to_str;

import lib::llvm::llvm;
import lib::llvm::llvm::ValueRef;
import lib::llvm::False;

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

const uint tag_index = 0x11u;
const uint tag_index_buckets = 0x12u;
const uint tag_index_buckets_bucket = 0x13u;
const uint tag_index_buckets_bucket_elt = 0x14u;
const uint tag_index_table = 0x15u;

const uint tag_meta_export = 0x16u;
const uint tag_meta_local = 0x17u;
const uint tag_meta_item = 0x18u;
const uint tag_meta_item_key = 0x19u;
const uint tag_meta_item_value = 0x20u;

// Type encoding

// Compact string representation for ty.t values. API ty_str & parse_from_str.
// Extra parameters are for converting to/from def_ids in the string rep.
// Whatever format you choose should not contain pipe characters.

type ty_abbrev = rec(uint pos, uint len, str s);

tag abbrev_ctxt {
    ac_no_abbrevs;
    ac_use_abbrevs(hashmap[ty::t, ty_abbrev]);
}

mod Encode {

    type ctxt = rec(
        fn(&ast::def_id) -> str ds,          // Def -> str Callback:
        ty::ctxt tcx,                        // The type context.
        abbrev_ctxt abbrevs
    );

    fn cx_uses_abbrevs(&@ctxt cx) -> bool {
        alt (cx.abbrevs) {
            case (ac_no_abbrevs)     { ret false; }
            case (ac_use_abbrevs(_)) { ret true; }
        }
    }

    fn ty_str(&@ctxt cx, &ty::t t) -> str {
        assert (!cx_uses_abbrevs(cx));
        auto sw = io::string_writer();
        enc_ty(sw.get_writer(), cx, t);
        ret sw.get_str();
    }

    fn enc_ty(&io::writer w, &@ctxt cx, &ty::t t) {
        alt (cx.abbrevs) {
            case (ac_no_abbrevs) {
                auto result_str;
                alt (cx.tcx.short_names_cache.find(t)) {
                    case (some(?s)) { result_str = s; }
                    case (none) {
                        auto sw = io::string_writer();
                        enc_sty(sw.get_writer(), cx, ty::struct(cx.tcx, t));
                        result_str = sw.get_str();
                        cx.tcx.short_names_cache.insert(t, result_str);
                    }
                }
                w.write_str(result_str);
            }
            case (ac_use_abbrevs(?abbrevs)) {
                alt (abbrevs.find(t)) {
                    case (some(?a)) {
                        w.write_str(a.s);
                        ret;
                    }
                    case (none) {
                        auto pos = w.get_buf_writer().tell();
                        auto ss = enc_sty(w, cx, ty::struct(cx.tcx, t));
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
                                      + uint::to_str(pos, 16u) + ":"
                                      + uint::to_str(len, 16u) + "#");
                            auto a = rec(pos=pos, len=len, s=s);
                            abbrevs.insert(t, a);
                        }
                        ret;
                    }
                }
            }
        }
    }

    fn enc_mt(&io::writer w, &@ctxt cx, &ty::mt mt) {
        alt (mt.mut) {
            case (ast::imm)       { }
            case (ast::mut)       { w.write_char('m'); }
            case (ast::maybe_mut) { w.write_char('?'); }
        }
        enc_ty(w, cx, mt.ty);
    }

    fn enc_sty(&io::writer w, &@ctxt cx, &ty::sty st) {
        alt (st) {
            case (ty::ty_nil) { w.write_char('n'); }
            case (ty::ty_bot) { w.write_char('z'); }
            case (ty::ty_bool) { w.write_char('b'); }
            case (ty::ty_int) { w.write_char('i'); }
            case (ty::ty_uint) { w.write_char('u'); }
            case (ty::ty_float) { w.write_char('l'); }
            case (ty::ty_machine(?mach)) {
                alt (mach) {
                    case (common::ty_u8) { w.write_str("Mb"); }
                    case (common::ty_u16) { w.write_str("Mw"); }
                    case (common::ty_u32) { w.write_str("Ml"); }
                    case (common::ty_u64) { w.write_str("Md"); }
                    case (common::ty_i8) { w.write_str("MB"); }
                    case (common::ty_i16) { w.write_str("MW"); }
                    case (common::ty_i32) { w.write_str("ML"); }
                    case (common::ty_i64) { w.write_str("MD"); }
                    case (common::ty_f32) { w.write_str("Mf"); }
                    case (common::ty_f64) { w.write_str("MF"); }
                }
            }
            case (ty::ty_char) {w.write_char('c');}
            case (ty::ty_str) {w.write_char('s');}
            case (ty::ty_istr) {w.write_char('S');}
            case (ty::ty_tag(?def,?tys)) {
                w.write_str("t[");
                w.write_str(cx.ds(def));
                w.write_char('|');
                for (ty::t t in tys) {
                    enc_ty(w, cx, t);
                }
                w.write_char(']');
            }
            case (ty::ty_box(?mt)) {w.write_char('@'); enc_mt(w, cx, mt); }
            case (ty::ty_ptr(?mt)) {w.write_char('*'); enc_mt(w, cx, mt); }
            case (ty::ty_vec(?mt)) {w.write_char('V'); enc_mt(w, cx, mt); }
            case (ty::ty_ivec(?mt)) {w.write_char('I'); enc_mt(w, cx, mt); }
            case (ty::ty_port(?t)) {w.write_char('P'); enc_ty(w, cx, t); }
            case (ty::ty_chan(?t)) {w.write_char('C'); enc_ty(w, cx, t); }
            case (ty::ty_tup(?mts)) {
                w.write_str("T[");
                for (ty::mt mt in mts) {
                    enc_mt(w, cx, mt);
                }
                w.write_char(']');
            }
            case (ty::ty_rec(?fields)) {
                w.write_str("R[");
                for (ty::field field in fields) {
                    w.write_str(field.ident);
                    w.write_char('=');
                    enc_mt(w, cx, field.mt);
                }
                w.write_char(']');
            }
            case (ty::ty_fn(?proto,?args,?out,?cf,?constrs)) {
                enc_proto(w, proto);
                enc_ty_fn(w, cx, args, out, cf, constrs);
            }
            case (ty::ty_native_fn(?abi,?args,?out)) {
                w.write_char('N');
                alt (abi) {
                    case (ast::native_abi_rust) { w.write_char('r'); }
                    case (ast::native_abi_rust_intrinsic) {
                        w.write_char('i');
                    }
                    case (ast::native_abi_cdecl) { w.write_char('c'); }
                    case (ast::native_abi_llvm) { w.write_char('l'); }
                }
                let vec[@constr] res_constrs = [];
                enc_ty_fn(w, cx, args, out, ast::return, res_constrs);
            }
            case (ty::ty_obj(?methods)) {
                w.write_str("O[");
                for (ty::method m in methods) {
                    enc_proto(w, m.proto);
                    w.write_str(m.ident);
                    enc_ty_fn(w, cx, m.inputs, m.output, m.cf, m.constrs);
                }
                w.write_char(']');
            }
            case (ty::ty_var(?id)) {
                w.write_char('X');
                w.write_str(common::istr(id));
            }
            case (ty::ty_native) {w.write_char('E');}
            case (ty::ty_param(?id)) {
                w.write_char('p');
                w.write_str(common::uistr(id));
            }
            case (ty::ty_type) {w.write_char('Y');}
            case (ty::ty_task) {w.write_char('a');}
        }
    }

    fn enc_proto(&io::writer w, ast::proto proto) {
        alt (proto) {
            case (ast::proto_iter) { w.write_char('W'); }
            case (ast::proto_fn) { w.write_char('F'); }
        }
    }

    fn enc_ty_fn(&io::writer w, &@ctxt cx, &vec[ty::arg] args, &ty::t out,
                 &ast::controlflow cf, &vec[@ast::constr] constrs) {
        w.write_char('[');
        for (ty::arg arg in args) {
            alt (arg.mode) {
                case (ty::mo_alias(?mut)) {
                    w.write_char('&');
                    if (mut) { w.write_char('m'); }
                }
                case (ty::mo_val) {}
            }
            enc_ty(w, cx, arg.ty);
        }
        w.write_char(']');
        alt (cf) {
            case (ast::noreturn) {
                w.write_char('!');
            }
            case (_) {
                enc_ty(w, cx, out);
            }
        }
        auto colon = true;
        for (@ast::constr c in constrs) {
            if (colon) { w.write_char(':'); colon = false; }
            else { w.write_char(','); }
            enc_constr(w, cx, c);
        }
    }

    fn enc_constr(&io::writer w, &@ctxt cx, &@ast::constr c) {
        w.write_str(path_to_str(c.node.path));
        w.write_char('(');
        // FIXME
        //   w.write_str(cx.ds(c.node.id));
        auto comma = false;
        for (@constr_arg a in c.node.args) {
            if (comma) {
                w.write_char(',');
            }
            else {
                comma = true;
            }
            alt (a.node) {
                case (carg_base) {
                    w.write_char('*');
                }
                case (carg_ident(?i)) {
                    w.write_uint(i);
                }
                case (carg_lit(?l)) {
                    w.write_str(lit_to_str(l));
                }
            }
        }
    }

}


// Returns a Plain Old LLVM String:
fn C_postr(&str s) -> ValueRef {
    ret llvm::LLVMConstString(str::buf(s), str::byte_len(s), False);
}


// Path table encoding

fn encode_name(&ebml::writer ebml_w, &str name) {
    ebml::start_tag(ebml_w, tag_paths_data_name);
    ebml_w.writer.write(str::bytes(name));
    ebml::end_tag(ebml_w);
}

fn encode_def_id(&ebml::writer ebml_w, &ast::def_id id) {
    ebml::start_tag(ebml_w, tag_def_id);
    ebml_w.writer.write(str::bytes(def_to_str(id)));
    ebml::end_tag(ebml_w);
}

fn encode_tag_variant_paths(&ebml::writer ebml_w,
                            &vec[ast::variant] variants,
                            &vec[str] path,
                            &mutable vec[tup(str, uint)] index) {
    for (ast::variant variant in variants) {
        add_to_index(ebml_w, path, index, variant.node.name);
        ebml::start_tag(ebml_w, tag_paths_data_item);
        encode_name(ebml_w, variant.node.name);
        encode_def_id(ebml_w, variant.node.id);
        ebml::end_tag(ebml_w);
    }
}

fn add_to_index(&ebml::writer ebml_w,
                &vec[str] path,
                &mutable vec[tup(str, uint)] index,
                &str name) {
    auto full_path = path + [name];
    index += [tup(str::connect(full_path, "::"), ebml_w.writer.tell())];
}

fn encode_native_module_item_paths(&ebml::writer ebml_w,
                                   &ast::native_mod nmod,
                                   &vec[str] path,
                                   &mutable vec[tup(str, uint)] index) {
    for (@ast::native_item nitem in nmod.items) {
        alt (nitem.node) {
            case (ast::native_item_ty(?id, ?did)) {
                add_to_index(ebml_w, path, index, id);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml::end_tag(ebml_w);
            }
            case (ast::native_item_fn(?id, _, _, _, ?did, _)) {
                add_to_index(ebml_w, path, index, id);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml::end_tag(ebml_w);
            }
        }
    }
}

fn encode_module_item_paths(&ebml::writer ebml_w,
                            &ast::_mod module,
                            &vec[str] path,
                            &mutable vec[tup(str, uint)] index) {
    for (@ast::item it in module.items) {
        if (!ast::is_exported(ast::item_ident(it), module)) {
            cont;
        }

        alt (it.node) {
            case (ast::item_const(?id, _, ?tps, ?did, ?ann)) {
                add_to_index(ebml_w, path, index, id);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml::end_tag(ebml_w);
            }
            case (ast::item_fn(?id, _, ?tps, ?did, ?ann)) {
                add_to_index(ebml_w, path, index, id);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml::end_tag(ebml_w);
            }
            case (ast::item_mod(?id, ?_mod, _, ?did)) {
                add_to_index(ebml_w, path, index, id);
                ebml::start_tag(ebml_w, tag_paths_data_mod);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                encode_module_item_paths(ebml_w, _mod, path + [id], index);
                ebml::end_tag(ebml_w);
            }
            case (ast::item_native_mod(?id, ?nmod, ?did)) {
                add_to_index(ebml_w, path, index, id);
                ebml::start_tag(ebml_w, tag_paths_data_mod);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                encode_native_module_item_paths(ebml_w, nmod, path + [id],
                                                index);
                ebml::end_tag(ebml_w);
            }
            case (ast::item_ty(?id, _, ?tps, ?did, ?ann)) {
                add_to_index(ebml_w, path, index, id);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml::end_tag(ebml_w);
            }
            case (ast::item_tag(?id, ?variants, ?tps, ?did, _)) {
                add_to_index(ebml_w, path, index, id);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml::end_tag(ebml_w);

                encode_tag_variant_paths(ebml_w, variants, path, index);
            }
            case (ast::item_obj(?id, _, ?tps, ?odid, ?ann)) {
                add_to_index(ebml_w, path, index, id);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, odid.ctor);
                ebml::end_tag(ebml_w);

                add_to_index(ebml_w, path, index, id);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, odid.ty);
                ebml::end_tag(ebml_w);
            }
        }
    }
}

fn encode_item_paths(&ebml::writer ebml_w, &@ast::crate crate)
        -> vec[tup(str, uint)] {
    let vec[tup(str, uint)] index = [];
    let vec[str] path = [];
    ebml::start_tag(ebml_w, tag_paths);
    encode_module_item_paths(ebml_w, crate.node.module, path, index);
    ebml::end_tag(ebml_w);
    ret index;
}


// Item info table encoding

fn encode_kind(&ebml::writer ebml_w, u8 c) {
    ebml::start_tag(ebml_w, tag_items_data_item_kind);
    ebml_w.writer.write([c]);
    ebml::end_tag(ebml_w);
}

fn def_to_str(&ast::def_id did) -> str {
    ret #fmt("%d:%d", did._0, did._1);
}

fn encode_type_param_count(&ebml::writer ebml_w, &vec[ast::ty_param] tps) {
    ebml::start_tag(ebml_w, tag_items_data_item_ty_param_count);
    ebml::write_vint(ebml_w.writer, vec::len[ast::ty_param](tps));
    ebml::end_tag(ebml_w);
}

fn encode_variant_id(&ebml::writer ebml_w, &ast::def_id vid) {
    ebml::start_tag(ebml_w, tag_items_data_item_variant);
    ebml_w.writer.write(str::bytes(def_to_str(vid)));
    ebml::end_tag(ebml_w);
}

fn encode_type(&@trans::crate_ctxt cx, &ebml::writer ebml_w, &ty::t typ) {
    ebml::start_tag(ebml_w, tag_items_data_item_type);

    auto f = def_to_str;
    auto ty_str_ctxt = @rec(ds=f, tcx=cx.tcx,
                            abbrevs=ac_use_abbrevs(cx.type_abbrevs));
    Encode::enc_ty(io::new_writer_(ebml_w.writer), ty_str_ctxt, typ);
    ebml::end_tag(ebml_w);
}

fn encode_symbol(&@trans::crate_ctxt cx, &ebml::writer ebml_w,
                 &ast::def_id did) {
    ebml::start_tag(ebml_w, tag_items_data_item_symbol);
    ebml_w.writer.write(str::bytes(cx.item_symbols.get(did)));
    ebml::end_tag(ebml_w);
}

fn encode_discriminant(&@trans::crate_ctxt cx, &ebml::writer ebml_w,
                       &ast::def_id did) {
    ebml::start_tag(ebml_w, tag_items_data_item_symbol);
    ebml_w.writer.write(str::bytes(cx.discrim_symbols.get(did)));
    ebml::end_tag(ebml_w);
}

fn encode_tag_id(&ebml::writer ebml_w, &ast::def_id id) {
    ebml::start_tag(ebml_w, tag_items_data_item_tag_id);
    ebml_w.writer.write(str::bytes(def_to_str(id)));
    ebml::end_tag(ebml_w);
}



fn encode_tag_variant_info(&@trans::crate_ctxt cx, &ebml::writer ebml_w,
                           &ast::def_id did, &vec[ast::variant] variants,
                           &mutable vec[tup(int, uint)] index,
                           &vec[ast::ty_param] ty_params) {
    for (ast::variant variant in variants) {
        index += [tup(variant.node.id._1, ebml_w.writer.tell())];

        ebml::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, variant.node.id);
        encode_kind(ebml_w, 'v' as u8);
        encode_tag_id(ebml_w, did);
        encode_type(cx, ebml_w, trans::node_ann_type(cx, variant.node.ann));
        if (vec::len[ast::variant_arg](variant.node.args) > 0u) {
            encode_symbol(cx, ebml_w, variant.node.id);
        }
        encode_discriminant(cx, ebml_w, variant.node.id);
        encode_type_param_count(ebml_w, ty_params);
        ebml::end_tag(ebml_w);
    }
}

fn encode_info_for_item(@trans::crate_ctxt cx, &ebml::writer ebml_w,
                        @ast::item item, &mutable vec[tup(int, uint)] index) {
    alt (item.node) {
        case (ast::item_const(_, _, _, ?did, ?ann)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'c' as u8);
            encode_type(cx, ebml_w, trans::node_ann_type(cx, ann));
            encode_symbol(cx, ebml_w, did);
            ebml::end_tag(ebml_w);
        }
        case (ast::item_fn(_, _, ?tps, ?did, ?ann)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'f' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, trans::node_ann_type(cx, ann));
            encode_symbol(cx, ebml_w, did);
            ebml::end_tag(ebml_w);
        }
        case (ast::item_mod(_, _, _, ?did)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'm' as u8);
            ebml::end_tag(ebml_w);
        }
        case (ast::item_native_mod(?id, _, ?did)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'n' as u8);
            ebml::end_tag(ebml_w);
        }
        case (ast::item_ty(?id, _, ?tps, ?did, ?ann)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'y' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, trans::node_ann_type(cx, ann));
            ebml::end_tag(ebml_w);
        }
        case (ast::item_tag(?id, ?variants, ?tps, ?did, ?ann)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 't' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, trans::node_ann_type(cx, ann));
            for (ast::variant v in variants) {
                encode_variant_id(ebml_w, v.node.id);
            }
            ebml::end_tag(ebml_w);

            encode_tag_variant_info(cx, ebml_w, did, variants, index, tps);
        }
        case (ast::item_obj(?id, _, ?tps, ?odid, ?ann)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, odid.ctor);
            encode_kind(ebml_w, 'o' as u8);
            encode_type_param_count(ebml_w, tps);
            auto fn_ty = trans::node_ann_type(cx, ann);
            encode_type(cx, ebml_w, fn_ty);
            encode_symbol(cx, ebml_w, odid.ctor);
            ebml::end_tag(ebml_w);

            index += [tup(odid.ty._1, ebml_w.writer.tell())];
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, odid.ty);
            encode_kind(ebml_w, 'y' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, ty::ty_fn_ret(cx.tcx, fn_ty));
            ebml::end_tag(ebml_w);
        }
    }
}

fn encode_info_for_native_item(&@trans::crate_ctxt cx, &ebml::writer ebml_w,
                               &@ast::native_item nitem) {
    ebml::start_tag(ebml_w, tag_items_data_item);
    alt (nitem.node) {
        case (ast::native_item_ty(_, ?did)) {
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'T' as u8);
            encode_type(cx, ebml_w, ty::mk_native(cx.tcx));
        }
        case (ast::native_item_fn(_, _, _, ?tps, ?did, ?ann)) {
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'F' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, trans::node_ann_type(cx, ann));
            encode_symbol(cx, ebml_w, did);
        }
    }
    ebml::end_tag(ebml_w);
}

fn encode_info_for_items(&@trans::crate_ctxt cx, &ebml::writer ebml_w)
        -> vec[tup(int, uint)] {
    let vec[tup(int, uint)] index = [];

    ebml::start_tag(ebml_w, tag_items_data);
    for each (@tup(ast::def_id, @ast::item) kvp in cx.items.items()) {
        index += [tup(kvp._0._1, ebml_w.writer.tell())];
        encode_info_for_item(cx, ebml_w, kvp._1, index);
    }
    for each (@tup(ast::def_id, @ast::native_item) kvp in
            cx.native_items.items()) {
        index += [tup(kvp._0._1, ebml_w.writer.tell())];
        encode_info_for_native_item(cx, ebml_w, kvp._1);
    }
    ebml::end_tag(ebml_w);

    ret index;
}


// Path and definition ID indexing

// djb's cdb hashes.

fn hash_def_num(&int def_num) -> uint {
    ret 177573u ^ (def_num as uint);
}

fn hash_path(&str s) -> uint {
    auto h = 5381u;
    for (u8 ch in str::bytes(s)) {
        h = ((h << 5u) + h) ^ (ch as uint);
    }
    ret h;
}

fn create_index[T](&vec[tup(T, uint)] index, fn(&T) -> uint hash_fn)
        -> vec[vec[tup(T, uint)]] {
    let vec[mutable vec[tup(T, uint)]] buckets = vec::empty_mut();
    for each (uint i in uint::range(0u, 256u)) {
        buckets += [mutable []];
    }

    for (tup(T, uint) elt in index) {
        auto h = hash_fn(elt._0);
        buckets.(h % 256u) += [elt];
    }

    ret vec::freeze(buckets);
}

fn encode_index[T](&ebml::writer ebml_w,
                   &vec[vec[tup(T, uint)]] buckets,
                   fn(&io::writer, &T) write_fn) {
    auto writer = io::new_writer_(ebml_w.writer);

    ebml::start_tag(ebml_w, tag_index);

    let vec[uint] bucket_locs = [];
    ebml::start_tag(ebml_w, tag_index_buckets);
    for (vec[tup(T, uint)] bucket in buckets) {
        bucket_locs += [ebml_w.writer.tell()];

        ebml::start_tag(ebml_w, tag_index_buckets_bucket);
        for (tup(T, uint) elt in bucket) {
            ebml::start_tag(ebml_w, tag_index_buckets_bucket_elt);
            writer.write_be_uint(elt._1, 4u);
            write_fn(writer, elt._0);
            ebml::end_tag(ebml_w);
        }
        ebml::end_tag(ebml_w);
    }
    ebml::end_tag(ebml_w);

    ebml::start_tag(ebml_w, tag_index_table);
    for (uint pos in bucket_locs) {
        writer.write_be_uint(pos, 4u);
    }
    ebml::end_tag(ebml_w);

    ebml::end_tag(ebml_w);
}

fn write_str(&io::writer writer, &str s) {
    writer.write_str(s);
}

fn write_int(&io::writer writer, &int n) {
    writer.write_be_uint(n as uint, 4u);
}


fn encode_meta_items(&ebml::writer ebml_w, &ast::crate crate) {

    fn encode_meta_item(&ebml::writer ebml_w, &ast::meta_item mi) {
        ebml::start_tag(ebml_w, tag_meta_item);
        ebml::start_tag(ebml_w, tag_meta_item_key);
        ebml_w.writer.write(str::bytes(mi.node.key));
        ebml::end_tag(ebml_w);
        ebml::start_tag(ebml_w, tag_meta_item_value);
        ebml_w.writer.write(str::bytes(mi.node.value));
        ebml::end_tag(ebml_w);
        ebml::end_tag(ebml_w);
    }

    ebml::start_tag(ebml_w, tag_meta_export);
    for each (@ast::meta_item mi in link::crate_export_metas(crate)) {
        encode_meta_item(ebml_w, *mi);
    }
    ebml::end_tag(ebml_w);

    ebml::start_tag(ebml_w, tag_meta_local);
    for each (@ast::meta_item mi in link::crate_local_metas(crate)) {
        encode_meta_item(ebml_w, *mi);
    }
    ebml::end_tag(ebml_w);
}

fn encode_metadata(&@trans::crate_ctxt cx, &@ast::crate crate)
        -> ValueRef {
    auto string_w = io::string_writer();
    auto buf_w = string_w.get_writer().get_buf_writer();
    auto ebml_w = ebml::create_writer(buf_w);

    // Encode the meta items
    encode_meta_items(ebml_w, *crate);

    // Encode and index the paths.
    ebml::start_tag(ebml_w, tag_paths);
    auto paths_index = encode_item_paths(ebml_w, crate);
    auto str_writer = write_str;
    auto path_hasher = hash_path;
    auto paths_buckets = create_index[str](paths_index, path_hasher);
    encode_index[str](ebml_w, paths_buckets, str_writer);
    ebml::end_tag(ebml_w);

    // Encode and index the items.
    ebml::start_tag(ebml_w, tag_items);
    auto items_index = encode_info_for_items(cx, ebml_w);
    auto int_writer = write_int;
    auto item_hasher = hash_def_num;
    auto items_buckets = create_index[int](items_index, item_hasher);
    encode_index[int](ebml_w, items_buckets, int_writer);
    ebml::end_tag(ebml_w);

    // Pad this, since something (LLVM, presumably) is cutting off the
    // remaining % 4 bytes.
    buf_w.write([0u8, 0u8, 0u8, 0u8]);

    ret C_postr(string_w.get_str());
}

fn write_metadata(&@trans::crate_ctxt cx, &@ast::crate crate) {
    if (!cx.sess.get_opts().shared) {
        ret;
    }

    auto llmeta = encode_metadata(cx, crate);
    auto llconst = trans::C_struct([llmeta]);
    auto llglobal = llvm::LLVMAddGlobal(cx.llmod, trans::val_ty(llconst),
                                       str::buf("rust_metadata"));
    llvm::LLVMSetInitializer(llglobal, llconst);
    llvm::LLVMSetSection(llglobal, str::buf(x86::get_meta_sect_name()));
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
