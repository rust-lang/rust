import std._str;
import std._uint;
import std._vec;
import std.ebml;
import std.io;
import std.option;

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
// (The second has to be authed pure.) Extra parameters are for converting
// to/from def_ids in the string rep. Whatever format you choose should not
// contain pipe characters.

mod Encode {

    type ctxt = rec(
        fn(ast.def_id) -> str ds    // Callback to translate defs to strs.
    );

    fn ty_str(@ctxt cx, ty.t t) -> str {
        ret sty_str(cx, ty.struct(t));
    }

    fn mt_str(@ctxt cx, &ty.mt mt) -> str {
        auto mut_str;
        alt (mt.mut) {
            case (ast.imm)       { mut_str = "";  }
            case (ast.mut)       { mut_str = "m"; }
            case (ast.maybe_mut) { mut_str = "?"; }
        }
        ret mut_str + ty_str(cx, mt.ty);
    }

    fn sty_str(@ctxt cx, ty.sty st) -> str {
        alt (st) {
            case (ty.ty_nil) {ret "n";}
            case (ty.ty_bool) {ret "b";}
            case (ty.ty_int) {ret "i";}
            case (ty.ty_uint) {ret "u";}
            case (ty.ty_float) {ret "l";}
            case (ty.ty_machine(?mach)) {
                alt (mach) {
                    case (common.ty_u8) {ret "Mb";}
                    case (common.ty_u16) {ret "Mw";}
                    case (common.ty_u32) {ret "Ml";}
                    case (common.ty_u64) {ret "Md";}
                    case (common.ty_i8) {ret "MB";}
                    case (common.ty_i16) {ret "MW";}
                    case (common.ty_i32) {ret "ML";}
                    case (common.ty_i64) {ret "MD";}
                    case (common.ty_f32) {ret "Mf";}
                    case (common.ty_f64) {ret "MF";}
                }
            }
            case (ty.ty_char) {ret "c";}
            case (ty.ty_str) {ret "s";}
            case (ty.ty_tag(?def,?tys)) { // TODO restore def_id
                auto acc = "t[" + cx.ds(def) + "|";
                for (ty.t t in tys) {acc += ty_str(cx, t);}
                ret acc + "]";
            }
            case (ty.ty_box(?mt)) {ret "@" + mt_str(cx, mt);}
            case (ty.ty_vec(?mt)) {ret "V" + mt_str(cx, mt);}
            case (ty.ty_port(?t)) {ret "P" + ty_str(cx, t);}
            case (ty.ty_chan(?t)) {ret "C" + ty_str(cx, t);}
            case (ty.ty_tup(?mts)) {
                auto acc = "T[";
                for (ty.mt mt in mts) {acc += mt_str(cx, mt);}
                ret acc + "]";
            }
            case (ty.ty_rec(?fields)) {
                auto acc = "R[";
                for (ty.field field in fields) {
                    acc += field.ident + "=";
                    acc += mt_str(cx, field.mt);
                }
                ret acc + "]";
            }
            case (ty.ty_fn(?proto,?args,?out)) {
                ret proto_str(proto) + ty_fn_str(cx, args, out);
            }
            case (ty.ty_native_fn(?abi,?args,?out)) {
                auto abistr;
                alt (abi) {
                    case (ast.native_abi_rust) {abistr = "r";}
                    case (ast.native_abi_cdecl) {abistr = "c";}
                    case (ast.native_abi_llvm) {abistr = "l";}
                }
                ret "N" + abistr + ty_fn_str(cx, args, out);
            }
            case (ty.ty_obj(?methods)) {
                auto acc = "O[";
                for (ty.method m in methods) {
                    acc += proto_str(m.proto);
                    acc += m.ident;
                    acc += ty_fn_str(cx, m.inputs, m.output);
                }
                ret acc + "]";
            }
            case (ty.ty_var(?id)) {ret "X" + common.istr(id);}
            case (ty.ty_native) {ret "E";}
            case (ty.ty_param(?id)) {ret "p" + common.uistr(id);}
            case (ty.ty_type) {ret "Y";}

            // These two don't appear in crate metadata, but are here because
            // `hash_ty()` uses this function.
            case (ty.ty_bound_param(?id)) {ret "o" + common.uistr(id);}
            case (ty.ty_local(?def)) {ret "L" + cx.ds(def);}
        }
    }

    fn proto_str(ast.proto proto) -> str {
        alt (proto) {
            case (ast.proto_iter) {ret "W";}
            case (ast.proto_fn) {ret "F";}
        }
    }

    fn ty_fn_str(@ctxt cx, vec[ty.arg] args, ty.t out) -> str {
        auto acc = "[";
        for (ty.arg arg in args) {
            if (arg.mode == ast.alias) {acc += "&";}
            acc += ty_str(cx, arg.ty);
        }
        ret acc + "]" + ty_str(cx, out);
    }

}


// Returns a Plain Old LLVM String, *without* the trailing zero byte.
fn C_postr(str s) -> ValueRef {
    ret llvm.LLVMConstString(_str.buf(s), _str.byte_len(s) - 1u, False);
}


// Path table encoding

fn encode_name(&ebml.writer ebml_w, str name) {
    ebml.start_tag(ebml_w, tag_paths_data_name);
    ebml_w.writer.write(_str.bytes(name));
    ebml.end_tag(ebml_w);
}

fn encode_def_id(&ebml.writer ebml_w, &ast.def_id id) {
    ebml.start_tag(ebml_w, tag_def_id);
    ebml_w.writer.write(_str.bytes(def_to_str(id)));
    ebml.end_tag(ebml_w);
}

fn encode_tag_variant_paths(&ebml.writer ebml_w,
                            vec[ast.variant] variants,
                            vec[str] path,
                            &mutable vec[tup(str, uint)] index) {
    for (ast.variant variant in variants) {
        add_to_index(ebml_w, path, index, variant.node.name);
        ebml.start_tag(ebml_w, tag_paths_data_item);
        encode_name(ebml_w, variant.node.name);
        encode_def_id(ebml_w, variant.node.id);
        ebml.end_tag(ebml_w);
    }
}

fn add_to_index(&ebml.writer ebml_w,
                vec[str] path,
                &mutable vec[tup(str, uint)] index,
                str name) {
    auto full_path = path + vec(name);
    index += vec(tup(_str.connect(full_path, "."), ebml_w.writer.tell()));
}

fn encode_native_module_item_paths(&ebml.writer ebml_w,
                                   &ast.native_mod nmod,
                                   vec[str] path,
                                   &mutable vec[tup(str, uint)] index) {
    for (@ast.native_item nitem in nmod.items) {
        alt (nitem.node) {
            case (ast.native_item_ty(?id, ?did)) {
                add_to_index(ebml_w, path, index, id);
                ebml.start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml.end_tag(ebml_w);
            }
            case (ast.native_item_fn(?id, _, _, _, ?did, _)) {
                add_to_index(ebml_w, path, index, id);
                ebml.start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml.end_tag(ebml_w);
            }
        }
    }
}

fn encode_module_item_paths(&ebml.writer ebml_w,
                            &ast._mod module,
                            vec[str] path,
                            &mutable vec[tup(str, uint)] index) {
    // TODO: only encode exported items
    for (@ast.item it in module.items) {
        alt (it.node) {
            case (ast.item_const(?id, _, ?tps, ?did, ?ann)) {
                add_to_index(ebml_w, path, index, id);
                ebml.start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml.end_tag(ebml_w);
            }
            case (ast.item_fn(?id, _, ?tps, ?did, ?ann)) {
                add_to_index(ebml_w, path, index, id);
                ebml.start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml.end_tag(ebml_w);
            }
            case (ast.item_mod(?id, ?_mod, ?did)) {
                add_to_index(ebml_w, path, index, id);
                ebml.start_tag(ebml_w, tag_paths_data_mod);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                encode_module_item_paths(ebml_w, _mod, path + vec(id), index);
                ebml.end_tag(ebml_w);
            }
            case (ast.item_native_mod(?id, ?nmod, ?did)) {
                add_to_index(ebml_w, path, index, id);
                ebml.start_tag(ebml_w, tag_paths_data_mod);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                encode_native_module_item_paths(ebml_w, nmod, path + vec(id),
                                                index);
                ebml.end_tag(ebml_w);
            }
            case (ast.item_ty(?id, _, ?tps, ?did, ?ann)) {
                add_to_index(ebml_w, path, index, id);
                ebml.start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml.end_tag(ebml_w);
            }
            case (ast.item_tag(?id, ?variants, ?tps, ?did, _)) {
                add_to_index(ebml_w, path, index, id);
                ebml.start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml.end_tag(ebml_w);

                encode_tag_variant_paths(ebml_w, variants, path, index);
            }
            case (ast.item_obj(?id, _, ?tps, ?odid, ?ann)) {
                add_to_index(ebml_w, path, index, id);
                ebml.start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, odid.ctor);
                encode_obj_type_id(ebml_w, odid.ty);
                ebml.end_tag(ebml_w);
            }
        }
    }
}

fn encode_item_paths(&ebml.writer ebml_w, @ast.crate crate)
        -> vec[tup(str, uint)] {
    let vec[tup(str, uint)] index = vec();
    let vec[str] path = vec();
    ebml.start_tag(ebml_w, tag_paths);
    encode_module_item_paths(ebml_w, crate.node.module, path, index);
    ebml.end_tag(ebml_w);
    ret index;
}


// Item info table encoding

fn encode_kind(&ebml.writer ebml_w, u8 c) {
    ebml.start_tag(ebml_w, tag_items_data_item_kind);
    ebml_w.writer.write(vec(c));
    ebml.end_tag(ebml_w);
}

fn def_to_str(ast.def_id did) -> str {
    ret #fmt("%d:%d", did._0, did._1);
}

fn encode_type_param_count(&ebml.writer ebml_w, vec[ast.ty_param] tps) {
    ebml.start_tag(ebml_w, tag_items_data_item_ty_param_count);
    ebml.write_vint(ebml_w.writer, _vec.len[ast.ty_param](tps));
    ebml.end_tag(ebml_w);
}

fn encode_variant_id(&ebml.writer ebml_w, ast.def_id vid) {
    ebml.start_tag(ebml_w, tag_items_data_item_variant);
    ebml_w.writer.write(_str.bytes(def_to_str(vid)));
    ebml.end_tag(ebml_w);
}

fn encode_type(&ebml.writer ebml_w, ty.t typ) {
    ebml.start_tag(ebml_w, tag_items_data_item_type);

    auto f = def_to_str;
    auto ty_str_ctxt = @rec(ds=f);
    ebml_w.writer.write(_str.bytes(Encode.ty_str(ty_str_ctxt, typ)));

    ebml.end_tag(ebml_w);
}

fn encode_symbol(@trans.crate_ctxt cx, &ebml.writer ebml_w, ast.def_id did) {
    ebml.start_tag(ebml_w, tag_items_data_item_symbol);
    ebml_w.writer.write(_str.bytes(cx.item_symbols.get(did)));
    ebml.end_tag(ebml_w);
}

fn encode_discriminant(@trans.crate_ctxt cx, &ebml.writer ebml_w,
                       ast.def_id did) {
    ebml.start_tag(ebml_w, tag_items_data_item_symbol);
    ebml_w.writer.write(_str.bytes(cx.discrim_symbols.get(did)));
    ebml.end_tag(ebml_w);
}

fn encode_tag_id(&ebml.writer ebml_w, &ast.def_id id) {
    ebml.start_tag(ebml_w, tag_items_data_item_tag_id);
    ebml_w.writer.write(_str.bytes(def_to_str(id)));
    ebml.end_tag(ebml_w);
}

fn encode_obj_type_id(&ebml.writer ebml_w, &ast.def_id id) {
    ebml.start_tag(ebml_w, tag_items_data_item_obj_type_id);
    ebml_w.writer.write(_str.bytes(def_to_str(id)));
    ebml.end_tag(ebml_w);
}


fn encode_tag_variant_info(@trans.crate_ctxt cx, &ebml.writer ebml_w,
                           ast.def_id did, vec[ast.variant] variants,
                           &mutable vec[tup(int, uint)] index,
                           vec[ast.ty_param] ty_params) {
    for (ast.variant variant in variants) {
        index += vec(tup(variant.node.id._1, ebml_w.writer.tell()));

        ebml.start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, variant.node.id);
        encode_kind(ebml_w, 'v' as u8);
        encode_tag_id(ebml_w, did);
        encode_type(ebml_w, trans.node_ann_type(cx, variant.node.ann));
        if (_vec.len[ast.variant_arg](variant.node.args) > 0u) {
            encode_symbol(cx, ebml_w, variant.node.id);
        }
        encode_discriminant(cx, ebml_w, variant.node.id);
        encode_type_param_count(ebml_w, ty_params);
        ebml.end_tag(ebml_w);
    }
}

fn encode_info_for_item(@trans.crate_ctxt cx, &ebml.writer ebml_w,
                        @ast.item item, &mutable vec[tup(int, uint)] index) {
    alt (item.node) {
        case (ast.item_const(_, _, _, ?did, ?ann)) {
            ebml.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'c' as u8);
            encode_type(ebml_w, trans.node_ann_type(cx, ann));
            encode_symbol(cx, ebml_w, did);
            ebml.end_tag(ebml_w);
        }
        case (ast.item_fn(_, _, ?tps, ?did, ?ann)) {
            ebml.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'f' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(ebml_w, trans.node_ann_type(cx, ann));
            encode_symbol(cx, ebml_w, did);
            ebml.end_tag(ebml_w);
        }
        case (ast.item_mod(_, _, ?did)) {
            ebml.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'm' as u8);
            ebml.end_tag(ebml_w);
        }
        case (ast.item_native_mod(_, _, ?did)) {
            ebml.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'n' as u8);
            ebml.end_tag(ebml_w);
        }
        case (ast.item_ty(?id, _, ?tps, ?did, ?ann)) {
            ebml.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'y' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(ebml_w, trans.node_ann_type(cx, ann));
            ebml.end_tag(ebml_w);
        }
        case (ast.item_tag(?id, ?variants, ?tps, ?did, ?ann)) {
            ebml.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 't' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(ebml_w, trans.node_ann_type(cx, ann));
            for (ast.variant v in variants) {
                encode_variant_id(ebml_w, v.node.id);
            }
            ebml.end_tag(ebml_w);

            encode_tag_variant_info(cx, ebml_w, did, variants, index, tps);
        }
        case (ast.item_obj(?id, _, ?tps, ?odid, ?ann)) {
            ebml.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, odid.ctor);
            encode_kind(ebml_w, 'o' as u8);
            encode_type_param_count(ebml_w, tps);
            auto fn_ty = trans.node_ann_type(cx, ann);
            encode_type(ebml_w, fn_ty);
            encode_symbol(cx, ebml_w, odid.ctor);
            ebml.end_tag(ebml_w);

            ebml.start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, odid.ty);
            encode_kind(ebml_w, 'y' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(ebml_w, ty.ty_fn_ret(fn_ty));
            ebml.end_tag(ebml_w);
        }
    }
}

fn encode_info_for_native_item(@trans.crate_ctxt cx, &ebml.writer ebml_w,
                               @ast.native_item nitem) {
    ebml.start_tag(ebml_w, tag_items_data_item);
    alt (nitem.node) {
        case (ast.native_item_ty(_, ?did)) {
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'T' as u8);
            encode_type(ebml_w, ty.mk_native(cx.tystore));
        }
        case (ast.native_item_fn(_, _, _, ?tps, ?did, ?ann)) {
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'F' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(ebml_w, trans.node_ann_type(cx, ann));
            encode_symbol(cx, ebml_w, did);
        }
    }
    ebml.end_tag(ebml_w);
}

fn encode_info_for_items(@trans.crate_ctxt cx, &ebml.writer ebml_w)
        -> vec[tup(int, uint)] {
    let vec[tup(int, uint)] index = vec();

    ebml.start_tag(ebml_w, tag_items_data);
    for each (@tup(ast.def_id, @ast.item) kvp in cx.items.items()) {
        index += vec(tup(kvp._0._1, ebml_w.writer.tell()));
        encode_info_for_item(cx, ebml_w, kvp._1, index);
    }
    for each (@tup(ast.def_id, @ast.native_item) kvp in
            cx.native_items.items()) {
        index += vec(tup(kvp._0._1, ebml_w.writer.tell()));
        encode_info_for_native_item(cx, ebml_w, kvp._1);
    }
    ebml.end_tag(ebml_w);

    ret index;
}


// Path and definition ID indexing

// djb's cdb hashes.

fn hash_def_num(&int def_num) -> uint {
    ret 177573u ^ (def_num as uint);
}

fn hash_path(&str s) -> uint {
    auto h = 5381u;
    for (u8 ch in _str.bytes(s)) {
        h = ((h << 5u) + h) ^ (ch as uint);
    }
    ret h;
}

fn create_index[T](vec[tup(T, uint)] index, fn(&T) -> uint hash_fn)
        -> vec[vec[tup(T, uint)]] {
    let vec[vec[tup(T, uint)]] buckets = vec();
    for each (uint i in _uint.range(0u, 256u)) {
        let vec[tup(T, uint)] bucket = vec();
        buckets += vec(bucket);
    }

    for (tup(T, uint) elt in index) {
        auto h = hash_fn(elt._0);
        buckets.(h % 256u) += vec(elt);
    }

    ret buckets;
}

fn encode_index[T](&ebml.writer ebml_w, vec[vec[tup(T, uint)]] buckets,
                          fn(io.writer, &T) write_fn) {
    auto writer = io.new_writer_(ebml_w.writer);

    ebml.start_tag(ebml_w, tag_index);

    let vec[uint] bucket_locs = vec();
    ebml.start_tag(ebml_w, tag_index_buckets);
    for (vec[tup(T, uint)] bucket in buckets) {
        bucket_locs += vec(ebml_w.writer.tell());

        ebml.start_tag(ebml_w, tag_index_buckets_bucket);
        for (tup(T, uint) elt in bucket) {
            ebml.start_tag(ebml_w, tag_index_buckets_bucket_elt);
            writer.write_be_uint(elt._1, 4u);
            write_fn(writer, elt._0);
            ebml.end_tag(ebml_w);
        }
        ebml.end_tag(ebml_w);
    }
    ebml.end_tag(ebml_w);

    ebml.start_tag(ebml_w, tag_index_table);
    for (uint pos in bucket_locs) {
        writer.write_be_uint(pos, 4u);
    }
    ebml.end_tag(ebml_w);

    ebml.end_tag(ebml_w);
}


fn write_str(io.writer writer, &str s) {
    writer.write_str(s);
}

fn write_int(io.writer writer, &int n) {
    writer.write_be_uint(n as uint, 4u);
}


fn encode_metadata(@trans.crate_ctxt cx, @ast.crate crate)
        -> ValueRef {
    auto string_w = io.string_writer();
    auto buf_w = string_w.get_writer().get_buf_writer();
    auto ebml_w = ebml.create_writer(buf_w);

    // Encode and index the paths.
    ebml.start_tag(ebml_w, tag_paths);
    auto paths_index = encode_item_paths(ebml_w, crate);
    auto str_writer = write_str;
    auto path_hasher = hash_path;
    auto paths_buckets = create_index[str](paths_index, path_hasher);
    encode_index[str](ebml_w, paths_buckets, str_writer);
    ebml.end_tag(ebml_w);

    // Encode and index the items.
    ebml.start_tag(ebml_w, tag_items);
    auto items_index = encode_info_for_items(cx, ebml_w);
    auto int_writer = write_int;
    auto item_hasher = hash_def_num;
    auto items_buckets = create_index[int](items_index, item_hasher);
    encode_index[int](ebml_w, items_buckets, int_writer);
    ebml.end_tag(ebml_w);

    // Pad this, since something (LLVM, presumably) is cutting off the
    // remaining % 4 bytes.
    buf_w.write(vec(0u8, 0u8, 0u8, 0u8));

    ret C_postr(string_w.get_str());
}

fn write_metadata(@trans.crate_ctxt cx, @ast.crate crate) {
    auto llmeta = encode_metadata(cx, crate);

    auto llconst = trans.C_struct(vec(llmeta));
    auto llglobal = llvm.LLVMAddGlobal(cx.llmod, trans.val_ty(llconst),
                                       _str.buf("rust_metadata"));
    llvm.LLVMSetInitializer(llglobal, llconst);
    llvm.LLVMSetSection(llglobal, _str.buf(x86.get_meta_sect_name()));
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
