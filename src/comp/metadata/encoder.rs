// Metadata encoding

import std::ivec;
import std::str;
import std::vec;
import std::uint;
import std::io;
import std::option;
import std::option::some;
import std::option::none;
import std::ebml;
import std::map;
import syntax::ast::*;
import common::*;
import middle::trans::crate_ctxt;
import middle::ty;
import middle::ty::node_id_to_monotype;
import front::attr;

export encode_metadata;
export encoded_ty;

type abbrev_map = map::hashmap[ty::t, tyencode::ty_abbrev];

type encode_ctxt = rec(@crate_ctxt ccx,
                       abbrev_map type_abbrevs);

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

fn encode_tag_variant_paths(&ebml::writer ebml_w, &variant[] variants,
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

fn encode_type_param_count(&ebml::writer ebml_w, &ty_param[] tps) {
    ebml::start_tag(ebml_w, tag_items_data_item_ty_param_count);
    ebml::write_vint(ebml_w.writer, ivec::len[ty_param](tps));
    ebml::end_tag(ebml_w);
}

fn encode_variant_id(&ebml::writer ebml_w, &def_id vid) {
    ebml::start_tag(ebml_w, tag_items_data_item_variant);
    ebml_w.writer.write(str::bytes(def_to_str(vid)));
    ebml::end_tag(ebml_w);
}

fn encode_type(&@encode_ctxt ecx, &ebml::writer ebml_w, &ty::t typ) {
    ebml::start_tag(ebml_w, tag_items_data_item_type);
    auto f = def_to_str;
    auto ty_str_ctxt =
        @rec(ds=f, tcx=ecx.ccx.tcx,
             abbrevs=tyencode::ac_use_abbrevs(ecx.type_abbrevs));
    tyencode::enc_ty(io::new_writer_(ebml_w.writer), ty_str_ctxt, typ);
    ebml::end_tag(ebml_w);
}

fn encode_symbol(&@encode_ctxt ecx, &ebml::writer ebml_w,
                 node_id id) {
    ebml::start_tag(ebml_w, tag_items_data_item_symbol);
    ebml_w.writer.write(str::bytes(ecx.ccx.item_symbols.get(id)));
    ebml::end_tag(ebml_w);
}

fn encode_discriminant(&@encode_ctxt ecx, &ebml::writer ebml_w,
                       node_id id) {
    ebml::start_tag(ebml_w, tag_items_data_item_symbol);
    ebml_w.writer.write(str::bytes(ecx.ccx.discrim_symbols.get(id)));
    ebml::end_tag(ebml_w);
}

fn encode_tag_id(&ebml::writer ebml_w, &def_id id) {
    ebml::start_tag(ebml_w, tag_items_data_item_tag_id);
    ebml_w.writer.write(str::bytes(def_to_str(id)));
    ebml::end_tag(ebml_w);
}

fn encode_tag_variant_info(&@encode_ctxt ecx, &ebml::writer ebml_w,
                           node_id id, &variant[] variants,
                           &mutable vec[tup(int, uint)] index,
                           &ty_param[] ty_params) {
    for (variant variant in variants) {
        index += [tup(variant.node.id, ebml_w.writer.tell())];
        ebml::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(variant.node.id));
        encode_kind(ebml_w, 'v' as u8);
        encode_tag_id(ebml_w, local_def(id));
        encode_type(ecx, ebml_w,
                    node_id_to_monotype(ecx.ccx.tcx, variant.node.id));
        if (ivec::len[variant_arg](variant.node.args) > 0u) {
            encode_symbol(ecx, ebml_w, variant.node.id);
        }
        encode_discriminant(ecx, ebml_w, variant.node.id);
        encode_type_param_count(ebml_w, ty_params);
        ebml::end_tag(ebml_w);
    }
}

fn encode_info_for_item(@encode_ctxt ecx, &ebml::writer ebml_w,
                        @item item, &mutable vec[tup(int, uint)] index) {
    alt (item.node) {
        case (item_const(_, _)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(item.id));
            encode_kind(ebml_w, 'c' as u8);
            encode_type(ecx, ebml_w,
                        node_id_to_monotype(ecx.ccx.tcx, item.id));
            encode_symbol(ecx, ebml_w, item.id);
            ebml::end_tag(ebml_w);
        }
        case (item_fn(?fd, ?tps)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(item.id));
            encode_kind(ebml_w, alt (fd.decl.purity) {
                                  case (pure_fn) { 'p' }
                                  case (impure_fn) { 'f' } } as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(ecx, ebml_w,
                        node_id_to_monotype(ecx.ccx.tcx, item.id));
            encode_symbol(ecx, ebml_w, item.id);
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
            encode_type(ecx, ebml_w,
                        node_id_to_monotype(ecx.ccx.tcx, item.id));
            ebml::end_tag(ebml_w);
        }
        case (item_tag(?variants, ?tps)) {
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(item.id));
            encode_kind(ebml_w, 't' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(ecx, ebml_w,
                        node_id_to_monotype(ecx.ccx.tcx, item.id));
            for (variant v in variants) {
                encode_variant_id(ebml_w, local_def(v.node.id));
            }
            ebml::end_tag(ebml_w);
            encode_tag_variant_info(ecx, ebml_w, item.id, variants, index,
                                    tps);
        }
        case (item_res(_, _, ?tps, ?ctor_id)) {
            auto fn_ty = node_id_to_monotype(ecx.ccx.tcx, ctor_id);

            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(ctor_id));
            encode_kind(ebml_w, 'y' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(ecx, ebml_w, ty::ty_fn_ret(ecx.ccx.tcx, fn_ty));
            encode_symbol(ecx, ebml_w, item.id);
            ebml::end_tag(ebml_w);

            index += [tup(ctor_id, ebml_w.writer.tell())];
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(ctor_id));
            encode_kind(ebml_w, 'f' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(ecx, ebml_w, fn_ty);
            encode_symbol(ecx, ebml_w, ctor_id);
            ebml::end_tag(ebml_w);
        }
        case (item_obj(_, ?tps, ?ctor_id)) {
            auto fn_ty = node_id_to_monotype(ecx.ccx.tcx, ctor_id);

            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(item.id));
            encode_kind(ebml_w, 'y' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(ecx, ebml_w, ty::ty_fn_ret(ecx.ccx.tcx, fn_ty));
            ebml::end_tag(ebml_w);

            index += [tup(ctor_id, ebml_w.writer.tell())];
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(ctor_id));
            encode_kind(ebml_w, 'f' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(ecx, ebml_w, fn_ty);
            encode_symbol(ecx, ebml_w, ctor_id);
            ebml::end_tag(ebml_w);
        }
    }
}

fn encode_info_for_native_item(&@encode_ctxt ecx, &ebml::writer ebml_w,
                               &@native_item nitem) {
    ebml::start_tag(ebml_w, tag_items_data_item);
    alt (nitem.node) {
        case (native_item_ty) {
            encode_def_id(ebml_w, local_def(nitem.id));
            encode_kind(ebml_w, 'T' as u8);
            encode_type(ecx, ebml_w,
                        ty::mk_native(ecx.ccx.tcx, local_def(nitem.id)));
        }
        case (native_item_fn(_, _, ?tps)) {
            encode_def_id(ebml_w, local_def(nitem.id));
            encode_kind(ebml_w, 'F' as u8);
            encode_type_param_count(ebml_w, tps);
            encode_type(ecx, ebml_w,
                        node_id_to_monotype(ecx.ccx.tcx, nitem.id));
            encode_symbol(ecx, ebml_w, nitem.id);
        }
    }
    ebml::end_tag(ebml_w);
}

fn encode_info_for_items(&@encode_ctxt ecx, &ebml::writer ebml_w) ->
   vec[tup(int, uint)] {
    let vec[tup(int, uint)] index = [];
    ebml::start_tag(ebml_w, tag_items_data);
    for each (@tup(node_id, middle::ast_map::ast_node) kvp in
              ecx.ccx.ast_map.items()) {
        alt (kvp._1) {
            case (middle::ast_map::node_item(?i)) {
                index += [tup(kvp._0, ebml_w.writer.tell())];
                encode_info_for_item(ecx, ebml_w, i, index);
            }
            case (middle::ast_map::node_native_item(?i)) {
                index += [tup(kvp._0, ebml_w.writer.tell())];
                encode_info_for_native_item(ecx, ebml_w, i);
            }
            case (_) {}
        }
    }
    ebml::end_tag(ebml_w);
    ret index;
}


// Path and definition ID indexing

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

fn encode_meta_item(&ebml::writer ebml_w, &meta_item mi) {
    alt (mi.node) {
        case (meta_word(?name)) {
            ebml::start_tag(ebml_w, tag_meta_item_word);
            ebml::start_tag(ebml_w, tag_meta_item_name);
            ebml_w.writer.write(str::bytes(name));
            ebml::end_tag(ebml_w);
            ebml::end_tag(ebml_w);
        }
        case (meta_name_value(?name, ?value)) {
            alt (value.node) {
                case (lit_str(?value, _)) {
                    ebml::start_tag(ebml_w, tag_meta_item_name_value);
                    ebml::start_tag(ebml_w, tag_meta_item_name);
                    ebml_w.writer.write(str::bytes(name));
                    ebml::end_tag(ebml_w);
                    ebml::start_tag(ebml_w, tag_meta_item_value);
                    ebml_w.writer.write(str::bytes(value));
                    ebml::end_tag(ebml_w);
                    ebml::end_tag(ebml_w);
                }
                case (_) { /* FIXME (#611) */ }
            }
        }
        case (meta_list(?name, ?items)) {
            ebml::start_tag(ebml_w, tag_meta_item_list);
            ebml::start_tag(ebml_w, tag_meta_item_name);
            ebml_w.writer.write(str::bytes(name));
            ebml::end_tag(ebml_w);
            for (@meta_item inner_item in items) {
                encode_meta_item(ebml_w, *inner_item);
            }
            ebml::end_tag(ebml_w);
        }
    }
}

fn encode_attributes(&ebml::writer ebml_w, &vec[attribute] attrs) {
    ebml::start_tag(ebml_w, tag_attributes);
    for (attribute attr in attrs) {
        ebml::start_tag(ebml_w, tag_attribute);
        encode_meta_item(ebml_w, attr.node.value);
        ebml::end_tag(ebml_w);
    }
    ebml::end_tag(ebml_w);
}

// So there's a special crate attribute called 'link' which defines the
// metadata that Rust cares about for linking crates. This attribute requires
// 'name' and 'vers' items, so if the user didn't provide them we will throw
// them in anyway with default values.
fn synthesize_crate_attrs(&@encode_ctxt ecx,
                          &@crate crate) -> vec[attribute] {

    fn synthesize_link_attr(&@encode_ctxt ecx, &(@meta_item)[] items)
            -> attribute {

        assert ecx.ccx.link_meta.name != "";
        assert ecx.ccx.link_meta.vers != "";

        auto name_item = attr::mk_name_value_item_str("name",
                                                      ecx.ccx.link_meta.name);
        auto vers_item = attr::mk_name_value_item_str("vers",
                                                      ecx.ccx.link_meta.vers);

        auto other_items = {
            auto tmp = attr::remove_meta_items_by_name(items, "name");
            attr::remove_meta_items_by_name(tmp, "vers")
        };

        auto meta_items = ~[name_item, vers_item] + other_items;
        auto link_item = attr::mk_list_item("link", meta_items);

        ret attr::mk_attr(link_item);
    }

    let vec[attribute] attrs = [];
    auto found_link_attr = false;
    for (attribute attr in crate.node.attrs) {
        attrs += if (attr::get_attr_name(attr) != "link") {
            [attr]
        } else {
            alt (attr.node.value.node) {
                case (meta_list(?n, ?l)) {
                    found_link_attr = true;
                    [synthesize_link_attr(ecx, l)]
                }
                case (_) { [attr] }
            }
        }
    }

    if (!found_link_attr) {
        attrs += [synthesize_link_attr(ecx, ~[])];
    }

    ret attrs;
}

fn encode_crate_deps(&ebml::writer ebml_w, &cstore::cstore cstore) {

    fn get_ordered_names(&cstore::cstore cstore) -> vec[str] {
        type hashkv = @tup(crate_num, cstore::crate_metadata);
        type numname = tup(crate_num, str);

        // Pull the cnums and names out of cstore
        let vec[mutable numname] pairs = [mutable];
        for each (hashkv hashkv in cstore.metas.items()) {
            pairs += [mutable tup(hashkv._0, hashkv._1.name)];
        }

        // Sort by cnum
        fn lteq(&numname kv1, &numname kv2) -> bool { kv1._0 <= kv2._0 }
        std::sort::quick_sort(lteq, pairs);

        // Sanity-check the crate numbers
        auto expected_cnum = 1;
        for (numname n in pairs) {
            assert n._0 == expected_cnum;
            expected_cnum += 1;
        }

        // Return just the names
        fn name(&numname kv) -> str { kv._1 }
        // mutable -> immutable hack for vec::map
        auto immpairs = vec::slice(pairs, 0u, vec::len(pairs));
        ret vec::map(name, immpairs);
    }

    // We're just going to write a list of crate names, with the assumption
    // that they are numbered 1 to n.
    // FIXME: This is not nearly enough to support correct versioning
    // but is enough to get transitive crate dependencies working.
    ebml::start_tag(ebml_w, tag_crate_deps);
    for (str cname in get_ordered_names(cstore)) {
        ebml::start_tag(ebml_w, tag_crate_dep);
        ebml_w.writer.write(str::bytes(cname));
        ebml::end_tag(ebml_w);
    }
    ebml::end_tag(ebml_w);
}

fn encode_metadata(&@crate_ctxt cx, &@crate crate) -> str {

    auto abbrevs = map::mk_hashmap(ty::hash_ty, ty::eq_ty);
    auto ecx = @rec(ccx = cx, type_abbrevs = abbrevs);

    auto string_w = io::string_writer();
    auto buf_w = string_w.get_writer().get_buf_writer();
    auto ebml_w = ebml::create_writer(buf_w);

    auto crate_attrs = synthesize_crate_attrs(ecx, crate);
    encode_attributes(ebml_w, crate_attrs);

    encode_crate_deps(ebml_w, cx.sess.get_cstore());

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
    auto items_index = encode_info_for_items(ecx, ebml_w);
    auto int_writer = write_int;
    auto item_hasher = hash_node_id;
    auto items_buckets = create_index[int](items_index, item_hasher);
    encode_index[int](ebml_w, items_buckets, int_writer);
    ebml::end_tag(ebml_w);
    // Pad this, since something (LLVM, presumably) is cutting off the
    // remaining % 4 bytes.

    buf_w.write([0u8, 0u8, 0u8, 0u8]);
    ret string_w.get_str();
}

// Get the encoded string for a type
fn encoded_ty(&ty::ctxt tcx, &ty::t t) -> str {
    auto cx = @rec(ds = def_to_str,
                   tcx = tcx,
                   abbrevs = tyencode::ac_no_abbrevs);
    auto sw = io::string_writer();
    tyencode::enc_ty(sw.get_writer(), cx, t);
    ret sw.get_str();
}


// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
