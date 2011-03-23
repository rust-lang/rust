import std._str;
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

const uint tag_paths_name = 0x03u;
const uint tag_paths_item = 0x04u;
const uint tag_paths_mod = 0x05u;

const uint tag_items_item = 0x06u;
const uint tag_items_def_id = 0x07u;
const uint tag_items_kind = 0x08u;
const uint tag_items_ty_param = 0x09u;
const uint tag_items_type = 0x0au;
const uint tag_items_symbol = 0x0bu;
const uint tag_items_variant = 0x0cu;
const uint tag_items_tag_id = 0x0du;

// Type encoding

// Compact string representation for ty.t values. API ty_str & parse_from_str.
// (The second has to be authed pure.) Extra parameters are for converting
// to/from def_ids in the string rep. Whatever format you choose should not
// contain pipe characters.

// Callback to translate defs to strs or back.
type def_str = fn(ast.def_id) -> str;

fn ty_str(@ty.t t, def_str ds) -> str {
    ret sty_str(t.struct, ds);
}

fn mt_str(&ty.mt mt, def_str ds) -> str {
    auto mut_str;
    alt (mt.mut) {
        case (ast.imm)       { mut_str = "";  }
        case (ast.mut)       { mut_str = "m"; }
        case (ast.maybe_mut) { mut_str = "?"; }
    }
    ret mut_str + ty_str(mt.ty, ds);
}

fn sty_str(ty.sty st, def_str ds) -> str {
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
            auto acc = "t[" + ds(def) + "|";
            for (@ty.t t in tys) {acc += ty_str(t, ds);}
            ret acc + "]";
        }
        case (ty.ty_box(?mt)) {ret "@" + mt_str(mt, ds);}
        case (ty.ty_vec(?mt)) {ret "V" + mt_str(mt, ds);}
        case (ty.ty_port(?t)) {ret "P" + ty_str(t, ds);}
        case (ty.ty_chan(?t)) {ret "C" + ty_str(t, ds);}
        case (ty.ty_tup(?mts)) {
            auto acc = "T[";
            for (ty.mt mt in mts) {acc += mt_str(mt, ds);}
            ret acc + "]";
        }
        case (ty.ty_rec(?fields)) {
            auto acc = "R[";
            for (ty.field field in fields) {
                acc += field.ident + "=";
                acc += mt_str(field.mt, ds);
            }
            ret acc + "]";
        }
        case (ty.ty_fn(?proto,?args,?out)) {
            ret proto_str(proto) + ty_fn_str(args, out, ds);
        }
        case (ty.ty_native_fn(?abi,?args,?out)) {
            auto abistr;
            alt (abi) {
                case (ast.native_abi_rust) {abistr = "r";}
                case (ast.native_abi_cdecl) {abistr = "c";}
            }
            ret "N" + abistr + ty_fn_str(args, out, ds);
        }
        case (ty.ty_obj(?methods)) {
            auto acc = "O[";
            for (ty.method m in methods) {
                acc += proto_str(m.proto);
                acc += m.ident;
                acc += ty_fn_str(m.inputs, m.output, ds);
            }
            ret acc + "]";
        }
        case (ty.ty_var(?id)) {ret "X" + common.istr(id);}
        case (ty.ty_native) {ret "E";}
        case (ty.ty_param(?def)) {ret "p" + ds(def);}
        // TODO (maybe?)   ty_type;
    }
}

fn proto_str(ast.proto proto) -> str {
    alt (proto) {
        case (ast.proto_iter) {ret "W";}
        case (ast.proto_fn) {ret "F";}
    }
}

fn ty_fn_str(vec[ty.arg] args, @ty.t out, def_str ds) -> str {
    auto acc = "[";
    for (ty.arg arg in args) {
        if (arg.mode == ast.alias) {acc += "&";}
        acc += ty_str(arg.ty, ds);
    }
    ret acc + "]" + ty_str(out, ds);
}


// Returns a Plain Old LLVM String.
fn C_postr(str s) -> ValueRef {
    ret llvm.LLVMConstString(_str.buf(s), _str.byte_len(s), False);
}


// Path table encoding

fn encode_name(&ebml.writer ebml_w, str name) {
    ebml.start_tag(ebml_w, tag_paths_name);
    ebml_w.writer.write(_str.bytes(name));
    ebml.end_tag(ebml_w);
}

fn encode_def_id(&ebml.writer ebml_w, &ast.def_id id) {
    ebml.start_tag(ebml_w, tag_items_def_id);
    ebml_w.writer.write(_str.bytes(def_to_str(id)));
    ebml.end_tag(ebml_w);
}

fn encode_tag_variant_paths(&ebml.writer ebml_w, vec[ast.variant] variants) {
    for (ast.variant variant in variants) {
        ebml.start_tag(ebml_w, tag_paths_item);
        encode_name(ebml_w, variant.name);
        encode_def_id(ebml_w, variant.id);
        ebml.end_tag(ebml_w);
    }
}

fn encode_native_module_item_paths(&ebml.writer ebml_w,
                                   &ast.native_mod nmod) {
    for (@ast.native_item nitem in nmod.items) {
        alt (nitem.node) {
            case (ast.native_item_ty(?id, ?did)) {
                ebml.start_tag(ebml_w, tag_paths_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml.end_tag(ebml_w);
            }
            case (ast.native_item_fn(?id, _, _, _, ?did, _)) {
                ebml.start_tag(ebml_w, tag_paths_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml.end_tag(ebml_w);
            }
        }
    }
}

fn encode_module_item_paths(&ebml.writer ebml_w, &ast._mod module) {
    // TODO: only encode exported items
    for (@ast.item it in module.items) {
        alt (it.node) {
            case (ast.item_const(?id, _, ?tps, ?did, ?ann)) {
                ebml.start_tag(ebml_w, tag_paths_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml.end_tag(ebml_w);
            }
            case (ast.item_fn(?id, _, ?tps, ?did, ?ann)) {
                ebml.start_tag(ebml_w, tag_paths_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml.end_tag(ebml_w);
            }
            case (ast.item_mod(?id, ?_mod, _)) {
                ebml.start_tag(ebml_w, tag_paths_mod);
                encode_name(ebml_w, id);
                encode_module_item_paths(ebml_w, _mod);
                ebml.end_tag(ebml_w);
            }
            case (ast.item_native_mod(?id, ?nmod, _)) {
                ebml.start_tag(ebml_w, tag_paths_mod);
                encode_name(ebml_w, id);
                encode_native_module_item_paths(ebml_w, nmod);
                ebml.end_tag(ebml_w);
            }
            case (ast.item_ty(?id, _, ?tps, ?did, ?ann)) {
                ebml.start_tag(ebml_w, tag_paths_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml.end_tag(ebml_w);
            }
            case (ast.item_tag(?id, ?variants, ?tps, ?did)) {
                ebml.start_tag(ebml_w, tag_paths_item);
                encode_name(ebml_w, id);
                encode_tag_variant_paths(ebml_w, variants);
                encode_def_id(ebml_w, did);
                ebml.end_tag(ebml_w);
            }
            case (ast.item_obj(?id, _, ?tps, ?did, ?ann)) {
                ebml.start_tag(ebml_w, tag_paths_item);
                encode_name(ebml_w, id);
                encode_def_id(ebml_w, did);
                ebml.end_tag(ebml_w);
            }
        }
    }
}

fn encode_item_paths(&ebml.writer ebml_w, @ast.crate crate) {
    ebml.start_tag(ebml_w, tag_paths);
    encode_module_item_paths(ebml_w, crate.node.module);
    ebml.end_tag(ebml_w);
}


// Item info table encoding

fn encode_kind(&ebml.writer ebml_w, u8 c) {
    ebml.start_tag(ebml_w, tag_items_kind);
    ebml_w.writer.write(vec(c));
    ebml.end_tag(ebml_w);
}

fn def_to_str(ast.def_id did) -> str {
    ret #fmt("%d:%d", did._0, did._1);
}

// TODO: We need to encode the "crate numbers" somewhere for diamond imports.
fn encode_type_params(&ebml.writer ebml_w, vec[ast.ty_param] tps) {
    for (ast.ty_param tp in tps) {
        ebml.start_tag(ebml_w, tag_items_ty_param);
        ebml_w.writer.write(_str.bytes(def_to_str(tp.id)));
        ebml.end_tag(ebml_w);
    }
}

fn encode_type(&ebml.writer ebml_w, @ty.t typ) {
    ebml.start_tag(ebml_w, tag_items_type);
    auto f = def_to_str;
    ebml_w.writer.write(_str.bytes(ty_str(typ, f)));
    ebml.end_tag(ebml_w);
}

fn encode_symbol(@trans.crate_ctxt cx, &ebml.writer ebml_w, ast.def_id did) {
    ebml.start_tag(ebml_w, tag_items_symbol);
    ebml_w.writer.write(_str.bytes(cx.item_symbols.get(did)));
    ebml.end_tag(ebml_w);
}

fn encode_discriminant(@trans.crate_ctxt cx, &ebml.writer ebml_w,
                       ast.def_id did) {
    ebml.start_tag(ebml_w, tag_items_symbol);
    ebml_w.writer.write(_str.bytes(cx.discrim_symbols.get(did)));
    ebml.end_tag(ebml_w);
}

fn encode_tag_id(&ebml.writer ebml_w, &ast.def_id id) {
    ebml.start_tag(ebml_w, tag_items_tag_id);
    ebml_w.writer.write(_str.bytes(def_to_str(id)));
    ebml.end_tag(ebml_w);
}


fn encode_tag_variant_info(@trans.crate_ctxt cx, &ebml.writer ebml_w,
                           ast.def_id did, vec[ast.variant] variants) {
    for (ast.variant variant in variants) {
        ebml.start_tag(ebml_w, tag_items_variant);
        encode_def_id(ebml_w, variant.id);
        encode_tag_id(ebml_w, did);
        encode_type(ebml_w, trans.node_ann_type(cx, variant.ann));
        if (_vec.len[ast.variant_arg](variant.args) > 0u) {
            encode_symbol(cx, ebml_w, variant.id);
        }
        encode_discriminant(cx, ebml_w, variant.id);
        ebml.end_tag(ebml_w);
    }
}

fn encode_info_for_item(@trans.crate_ctxt cx, &ebml.writer ebml_w,
                        @ast.item item) {
    alt (item.node) {
        case (ast.item_const(_, _, _, ?did, ?ann)) {
            ebml.start_tag(ebml_w, tag_items_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'c' as u8);
            encode_type(ebml_w, trans.node_ann_type(cx, ann));
            encode_symbol(cx, ebml_w, did);
            ebml.end_tag(ebml_w);
        }
        case (ast.item_fn(_, _, ?tps, ?did, ?ann)) {
            ebml.start_tag(ebml_w, tag_items_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'f' as u8);
            encode_type_params(ebml_w, tps);
            encode_type(ebml_w, trans.node_ann_type(cx, ann));
            encode_symbol(cx, ebml_w, did);
            ebml.end_tag(ebml_w);
        }
        case (ast.item_mod(_, _, _)) {
            // nothing to do
        }
        case (ast.item_native_mod(_, _, _)) {
            // nothing to do
        }
        case (ast.item_ty(?id, _, ?tps, ?did, ?ann)) {
            ebml.start_tag(ebml_w, tag_items_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'y' as u8);
            encode_type_params(ebml_w, tps);
            encode_type(ebml_w, trans.node_ann_type(cx, ann));
            ebml.end_tag(ebml_w);
        }
        case (ast.item_tag(?id, ?variants, ?tps, ?did)) {
            ebml.start_tag(ebml_w, tag_items_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 't' as u8);
            encode_type_params(ebml_w, tps);
            ebml.end_tag(ebml_w);

            encode_tag_variant_info(cx, ebml_w, did, variants);
        }
        case (ast.item_obj(?id, _, ?tps, ?did, ?ann)) {
            ebml.start_tag(ebml_w, tag_items_item);
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'o' as u8);
            encode_type_params(ebml_w, tps);
            encode_type(ebml_w, trans.node_ann_type(cx, ann));
            encode_symbol(cx, ebml_w, did);
            ebml.end_tag(ebml_w);
        }
    }
}

fn encode_info_for_native_item(@trans.crate_ctxt cx, &ebml.writer ebml_w,
                               @ast.native_item nitem) {
    ebml.start_tag(ebml_w, tag_items_item);
    alt (nitem.node) {
        case (ast.native_item_ty(_, ?did)) {
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'T' as u8);
        }
        case (ast.native_item_fn(_, _, _, ?tps, ?did, ?ann)) {
            encode_def_id(ebml_w, did);
            encode_kind(ebml_w, 'F' as u8);
            encode_type_params(ebml_w, tps);
            encode_type(ebml_w, trans.node_ann_type(cx, ann));
        }
    }
    ebml.end_tag(ebml_w);
}

fn encode_info_for_items(@trans.crate_ctxt cx, &ebml.writer ebml_w) {
    ebml.start_tag(ebml_w, tag_items);
    for each (@tup(ast.def_id, @ast.item) kvp in cx.items.items()) {
        encode_info_for_item(cx, ebml_w, kvp._1);
    }
    for each (@tup(ast.def_id, @ast.native_item) kvp in
            cx.native_items.items()) {
        encode_info_for_native_item(cx, ebml_w, kvp._1);
    }
    ebml.end_tag(ebml_w);
}


fn encode_metadata(@trans.crate_ctxt cx, @ast.crate crate) -> ValueRef {
    auto string_w = io.string_writer();
    auto buf_w = string_w.get_writer().get_buf_writer();
    auto ebml_w = ebml.create_writer(buf_w);

    encode_item_paths(ebml_w, crate);
    encode_info_for_items(cx, ebml_w);

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

