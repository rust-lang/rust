
import std::str;
import std::uint;
import std::vec;
import std::map::hashmap;
import std::ebml;
import std::io;
import std::option;
import std::option::some;
import std::option::none;
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

export ac_no_abbrevs;
export def_to_str;
export encode;
export hash_def_id;
export hash_path;
export tag_def_id;
export tag_index;
export tag_index_table;
export tag_index_buckets;
export tag_index_buckets_bucket;
export tag_index_buckets_bucket_elt;
export tag_items;
export tag_items_data_item_kind;
export tag_items_data_item_symbol;
export tag_items_data_item_tag_id;
export tag_items_data_item_type;
export tag_items_data_item_ty_param_count;
export tag_items_data_item_variant;
export tag_meta_export;
export tag_meta_item;
export tag_meta_item_key;
export tag_meta_item_value;
export tag_paths;
export ty_abbrev;
export write_metadata;

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

fn encode_def_id(&ebml::writer ebml_w, &def_id id) {
    ebml::start_tag(ebml_w, tag_def_id);
    ebml_w.writer.write(str::bytes(def_to_str(id)));
    ebml::end_tag(ebml_w);
}

fn encode_tag_variant_paths(&ebml::writer ebml_w, &vec[variant] variants,
                            &vec[str] path,
                            &mutable vec[tup(str, uint)] index) {
    for (variant variant in variants) {
        add_to_index(ebml_w, path, index, variant.node.name);
        ebml::start_tag(ebml_w, tag_paths_data_item);
        encode_name(ebml_w, variant.node.name);
        encode_def_id(ebml_w, local_def(variant.node.id));
        ebml::end_tag(ebml_w);
    }
}

fn add_to_index(&ebml::writer ebml_w, &vec[str] path,
                &mutable vec[tup(str, uint)] index, &str name) {
    auto full_path = path + [name];
    index += [tup(str::connect(full_path, "::"), ebml_w.writer.tell())];
}

fn encode_native_module_item_paths(&ebml::writer ebml_w,
                                   &native_mod nmod, &vec[str] path,
                                   &mutable vec[tup(str, uint)] index) {
    for (@native_item nitem in nmod.items) {
        add_to_index(ebml_w, path, index, nitem.ident);
        ebml::start_tag(ebml_w, tag_paths_data_item);
        encode_name(ebml_w, nitem.ident);
        encode_def_id(ebml_w, local_def(nitem.id));
        ebml::end_tag(ebml_w);
    }
}

fn encode_module_item_paths(&ebml::writer ebml_w, &_mod module,
                            &vec[str] path,
                            &mutable vec[tup(str, uint)] index) {
    for (@item it in module.items) {
        if (!is_exported(it.ident, module)) { cont; }
        alt (it.node) {
            case (item_const(_, _)) {
                add_to_index(ebml_w, path, index, it.ident);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, it.ident);
                encode_def_id(ebml_w, local_def(it.id));
                ebml::end_tag(ebml_w);
            }
            case (item_fn(_, ?tps)) {
                add_to_index(ebml_w, path, index, it.ident);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, it.ident);
                encode_def_id(ebml_w, local_def(it.id));
                ebml::end_tag(ebml_w);
            }
            case (item_mod(?_mod)) {
                add_to_index(ebml_w, path, index, it.ident);
                ebml::start_tag(ebml_w, tag_paths_data_mod);
                encode_name(ebml_w, it.ident);
                encode_def_id(ebml_w, local_def(it.id));
                encode_module_item_paths(ebml_w, _mod, path + [it.ident],
                                         index);
                ebml::end_tag(ebml_w);
            }
            case (item_native_mod(?nmod)) {
                add_to_index(ebml_w, path, index, it.ident);
                ebml::start_tag(ebml_w, tag_paths_data_mod);
                encode_name(ebml_w, it.ident);
                encode_def_id(ebml_w, local_def(it.id));
                encode_native_module_item_paths(ebml_w, nmod,
                                                path + [it.ident], index);
                ebml::end_tag(ebml_w);
            }
            case (item_ty(_, ?tps)) {
                add_to_index(ebml_w, path, index, it.ident);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, it.ident);
                encode_def_id(ebml_w, local_def(it.id));
                ebml::end_tag(ebml_w);
            }
            case (item_res(_, _, ?tps, ?ctor_id)) {
                add_to_index(ebml_w, path, index, it.ident);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, it.ident);
                encode_def_id(ebml_w, local_def(ctor_id));
                ebml::end_tag(ebml_w);
                add_to_index(ebml_w, path, index, it.ident);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, it.ident);
                encode_def_id(ebml_w, local_def(it.id));
                ebml::end_tag(ebml_w);
            }
            case (item_tag(?variants, ?tps)) {
                add_to_index(ebml_w, path, index, it.ident);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, it.ident);
                encode_def_id(ebml_w, local_def(it.id));
                ebml::end_tag(ebml_w);
                encode_tag_variant_paths(ebml_w, variants, path, index);
            }
            case (item_obj(_, ?tps, ?ctor_id)) {
                add_to_index(ebml_w, path, index, it.ident);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, it.ident);
                encode_def_id(ebml_w, local_def(ctor_id));
                ebml::end_tag(ebml_w);
                add_to_index(ebml_w, path, index, it.ident);
                ebml::start_tag(ebml_w, tag_paths_data_item);
                encode_name(ebml_w, it.ident);
                encode_def_id(ebml_w, local_def(it.id));
                ebml::end_tag(ebml_w);
            }
        }
    }
}

fn encode_item_paths(&ebml::writer ebml_w, &@crate crate) ->
   vec[tup(str, uint)] {
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

fn def_to_str(&def_id did) -> str { ret #fmt("%d:%d", did._0, did._1); }

fn encode_type_param_count(&ebml::writer ebml_w, &vec[ty_param] tps) {
    ebml::start_tag(ebml_w, tag_items_data_item_ty_param_count);
    ebml::write_vint(ebml_w.writer, vec::len[ty_param](tps));
    ebml::end_tag(ebml_w);
}

fn encode_variant_id(&ebml::writer ebml_w, &def_id vid) {
    ebml::start_tag(ebml_w, tag_items_data_item_variant);
    ebml_w.writer.write(str::bytes(def_to_str(vid)));
    ebml::end_tag(ebml_w);
}

fn encode_type(&@trans::crate_ctxt cx, &ebml::writer ebml_w, &ty::t typ) {
    ebml::start_tag(ebml_w, tag_items_data_item_type);
    auto f = def_to_str;
    auto ty_str_ctxt =
        @rec(ds=f, tcx=cx.tcx, abbrevs=tyencode::ac_use_abbrevs(cx.type_abbrevs));
    tyencode::enc_ty(io::new_writer_(ebml_w.writer), ty_str_ctxt, typ);
    ebml::end_tag(ebml_w);
}

fn encode_symbol(&@trans::crate_ctxt cx, &ebml::writer ebml_w,
                 node_id id) {
    ebml::start_tag(ebml_w, tag_items_data_item_symbol);
    ebml_w.writer.write(str::bytes(cx.item_symbols.get(id)));
    ebml::end_tag(ebml_w);
}

fn encode_discriminant(&@trans::crate_ctxt cx, &ebml::writer ebml_w,
                       node_id id) {
    ebml::start_tag(ebml_w, tag_items_data_item_symbol);
    ebml_w.writer.write(str::bytes(cx.discrim_symbols.get(id)));
    ebml::end_tag(ebml_w);
}

fn encode_tag_id(&ebml::writer ebml_w, &def_id id) {
    ebml::start_tag(ebml_w, tag_items_data_item_tag_id);
    ebml_w.writer.write(str::bytes(def_to_str(id)));
    ebml::end_tag(ebml_w);
}

fn encode_tag_variant_info(&@trans::crate_ctxt cx, &ebml::writer ebml_w,
                           node_id id, &vec[variant] variants,
                           &mutable vec[tup(int, uint)] index,
                           &vec[ty_param] ty_params) {
    for (variant variant in variants) {
        index += [tup(variant.node.id, ebml_w.writer.tell())];
        ebml::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(variant.node.id));
        encode_kind(ebml_w, 'v' as u8);
        encode_tag_id(ebml_w, local_def(id));
        encode_type(cx, ebml_w, trans::node_id_type(cx, variant.node.id));
        if (vec::len[variant_arg](variant.node.args) > 0u) {
            encode_symbol(cx, ebml_w, variant.node.id);
        }
        encode_discriminant(cx, ebml_w, variant.node.id);
        encode_type_param_count(ebml_w, ty_params);
        ebml::end_tag(ebml_w);
    }
}

fn encode_info_for_item(@trans::crate_ctxt cx, &ebml::writer ebml_w,
                        @item item, &mutable vec[tup(int, uint)] index) {
    alt (item.node) {
        case (item_const(_, _)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(item.id));
            encode_kind(ebml_w, 'c' as u8);
            encode_type(cx, ebml_w, trans::node_id_type(cx, item.id));
            encode_symbol(cx, ebml_w, item.id);
            ebml::end_tag(ebml_w);
        }
        case (item_fn(?fd, ?tps)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(item.id));
            encode_kind(ebml_w, alt (fd.decl.purity) {
                                  case (pure_fn) { 'p' }
                                  case (impure_fn) { 'f' } } as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, trans::node_id_type(cx, item.id));
            encode_symbol(cx, ebml_w, item.id);
            ebml::end_tag(ebml_w);
        }
        case (item_mod(_)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(item.id));
            encode_kind(ebml_w, 'm' as u8);
            ebml::end_tag(ebml_w);
        }
        case (item_native_mod(_)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(item.id));
            encode_kind(ebml_w, 'n' as u8);
            ebml::end_tag(ebml_w);
        }
        case (item_ty(_, ?tps)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(item.id));
            encode_kind(ebml_w, 'y' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, trans::node_id_type(cx, item.id));
            ebml::end_tag(ebml_w);
        }
        case (item_tag(?variants, ?tps)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(item.id));
            encode_kind(ebml_w, 't' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, trans::node_id_type(cx, item.id));
            for (variant v in variants) {
                encode_variant_id(ebml_w, local_def(v.node.id));
            }
            ebml::end_tag(ebml_w);
            encode_tag_variant_info(cx, ebml_w, item.id, variants, index,
                                    tps);
        }
        case (item_res(_, _, ?tps, ?ctor_id)) {
            auto fn_ty = trans::node_id_type(cx, item.id);

            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(item.id));
            encode_kind(ebml_w, 'y' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, ty::ty_fn_ret(cx.tcx, fn_ty));
            ebml::end_tag(ebml_w);

            index += [tup(ctor_id, ebml_w.writer.tell())];
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(ctor_id));
            encode_kind(ebml_w, 'f' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, fn_ty);
            encode_symbol(cx, ebml_w, ctor_id);
            ebml::end_tag(ebml_w);
        }
        case (item_obj(_, ?tps, ?ctor_id)) {
            auto fn_ty = trans::node_id_type(cx, ctor_id);

            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(item.id));
            encode_kind(ebml_w, 'y' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, ty::ty_fn_ret(cx.tcx, fn_ty));
            ebml::end_tag(ebml_w);

            index += [tup(ctor_id, ebml_w.writer.tell())];
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(ctor_id));
            encode_kind(ebml_w, 'f' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, fn_ty);
            encode_symbol(cx, ebml_w, ctor_id);
            ebml::end_tag(ebml_w);
        }
    }
}

fn encode_info_for_native_item(&@trans::crate_ctxt cx, &ebml::writer ebml_w,
                               &@native_item nitem) {
    ebml::start_tag(ebml_w, tag_items_data_item);
    alt (nitem.node) {
        case (native_item_ty) {
            encode_def_id(ebml_w, local_def(nitem.id));
            encode_kind(ebml_w, 'T' as u8);
            encode_type(cx, ebml_w, ty::mk_native(cx.tcx));
        }
        case (native_item_fn(_, _, ?tps)) {
            encode_def_id(ebml_w, local_def(nitem.id));
            encode_kind(ebml_w, 'F' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(cx, ebml_w, trans::node_id_type(cx, nitem.id));
            encode_symbol(cx, ebml_w, nitem.id);
        }
    }
    ebml::end_tag(ebml_w);
}

fn encode_info_for_items(&@trans::crate_ctxt cx, &ebml::writer ebml_w) ->
   vec[tup(int, uint)] {
    let vec[tup(int, uint)] index = [];
    ebml::start_tag(ebml_w, tag_items_data);
    for each (@tup(node_id, middle::ast_map::ast_node) kvp in cx.ast_map.items()) {
        alt (kvp._1) {
            case (middle::ast_map::node_item(?i)) {
                index += [tup(kvp._0, ebml_w.writer.tell())];
                encode_info_for_item(cx, ebml_w, i, index);
            }
            case (middle::ast_map::node_native_item(?i)) {
                index += [tup(kvp._0, ebml_w.writer.tell())];
                encode_info_for_native_item(cx, ebml_w, i);
            }
            case (_) {}
        }
    }
    ebml::end_tag(ebml_w);
    ret index;
}


// Path and definition ID indexing

// djb's cdb hashes.
fn hash_def_id(&int def_id) -> uint { ret 177573u ^ (def_id as uint); }

fn hash_path(&str s) -> uint {
    auto h = 5381u;
    for (u8 ch in str::bytes(s)) { h = (h << 5u) + h ^ (ch as uint); }
    ret h;
}

fn create_index[T](&vec[tup(T, uint)] index, fn(&T) -> uint  hash_fn) ->
   vec[vec[tup(T, uint)]] {
    let vec[mutable vec[tup(T, uint)]] buckets = vec::empty_mut();
    for each (uint i in uint::range(0u, 256u)) { buckets += [mutable []]; }
    for (tup(T, uint) elt in index) {
        auto h = hash_fn(elt._0);
        buckets.(h % 256u) += [elt];
    }
    ret vec::freeze(buckets);
}

fn encode_index[T](&ebml::writer ebml_w, &vec[vec[tup(T, uint)]] buckets,
                   fn(&io::writer, &T)  write_fn) {
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
    for (uint pos in bucket_locs) { writer.write_be_uint(pos, 4u); }
    ebml::end_tag(ebml_w);
    ebml::end_tag(ebml_w);
}

fn write_str(&io::writer writer, &str s) { writer.write_str(s); }

fn write_int(&io::writer writer, &int n) {
    writer.write_be_uint(n as uint, 4u);
}

fn encode_meta_items(&ebml::writer ebml_w, &crate crate) {
    fn encode_meta_item(&ebml::writer ebml_w, &meta_item mi) {
        // FIXME (#487): Support all forms of meta item
        ebml::start_tag(ebml_w, tag_meta_item);
        alt (mi.node) {
            case (meta_key_value(?key, ?value)) {
                ebml::start_tag(ebml_w, tag_meta_item_key);
                ebml_w.writer.write(str::bytes(key));
                ebml::end_tag(ebml_w);
                ebml::start_tag(ebml_w, tag_meta_item_value);
                ebml_w.writer.write(str::bytes(value));
                ebml::end_tag(ebml_w);
            }
            case (_) {
                log_err "unimplemented meta_item type";
            }
        }
        ebml::end_tag(ebml_w);
    }
    ebml::start_tag(ebml_w, tag_meta_export);
    for each (@meta_item mi in link::crate_export_metas(crate)) {
        encode_meta_item(ebml_w, *mi);
    }
    ebml::end_tag(ebml_w);
    ebml::start_tag(ebml_w, tag_meta_local);
    for each (@meta_item mi in link::crate_local_metas(crate)) {
        encode_meta_item(ebml_w, *mi);
    }
    ebml::end_tag(ebml_w);
}

fn encode_metadata(&@trans::crate_ctxt cx, &@crate crate) -> ValueRef {
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
    auto item_hasher = hash_def_id;
    auto items_buckets = create_index[int](items_index, item_hasher);
    encode_index[int](ebml_w, items_buckets, int_writer);
    ebml::end_tag(ebml_w);
    // Pad this, since something (LLVM, presumably) is cutting off the
    // remaining % 4 bytes.

    buf_w.write([0u8, 0u8, 0u8, 0u8]);
    ret C_postr(string_w.get_str());
}

fn write_metadata(&@trans::crate_ctxt cx, &@crate crate) {
    if (!cx.sess.get_opts().shared) { ret; }
    auto llmeta = encode_metadata(cx, crate);
    auto llconst = trans::C_struct([llmeta]);
    auto llglobal =
        llvm::LLVMAddGlobal(cx.llmod, trans::val_ty(llconst),
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
