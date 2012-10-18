// Metadata encoding

use util::ppaux::ty_to_str;

use std::{ebml, map};
use std::map::HashMap;
use io::WriterUtil;
use ebml::Serializer;
use syntax::ast::*;
use syntax::print::pprust;
use syntax::{ast_util, visit};
use syntax::ast_util::*;
use common::*;
use middle::ty;
use middle::ty::node_id_to_type;
use middle::resolve;
use syntax::ast_map;
use syntax::attr;
use str::to_bytes;
use syntax::ast;
use syntax::diagnostic::span_handler;

use hash::{Hash, HashUtil};
use to_bytes::IterBytes;

export encode_parms;
export encode_metadata;
export encoded_ty;
export reachable;
export encode_inlined_item;
export metadata_encoding_version;

// used by astencode:
export def_to_str;
export encode_ctxt;
export write_type;
export write_vstore;
export encode_def_id;

type abbrev_map = map::HashMap<ty::t, tyencode::ty_abbrev>;

type encode_inlined_item = fn@(ecx: @encode_ctxt,
                               ebml_w: ebml::Serializer,
                               path: ast_map::path,
                               ii: ast::inlined_item);

type encode_parms = {
    diag: span_handler,
    tcx: ty::ctxt,
    reachable: HashMap<ast::node_id, ()>,
    reexports2: middle::resolve::ExportMap2,
    item_symbols: HashMap<ast::node_id, ~str>,
    discrim_symbols: HashMap<ast::node_id, ~str>,
    link_meta: link_meta,
    cstore: cstore::CStore,
    encode_inlined_item: encode_inlined_item
};

type stats = {
    mut inline_bytes: uint,
    mut attr_bytes: uint,
    mut dep_bytes: uint,
    mut item_bytes: uint,
    mut index_bytes: uint,
    mut zero_bytes: uint,
    mut total_bytes: uint,

    mut n_inlines: uint
};

enum encode_ctxt = {
    diag: span_handler,
    tcx: ty::ctxt,
    stats: stats,
    reachable: HashMap<ast::node_id, ()>,
    reexports2: middle::resolve::ExportMap2,
    item_symbols: HashMap<ast::node_id, ~str>,
    discrim_symbols: HashMap<ast::node_id, ~str>,
    link_meta: link_meta,
    cstore: cstore::CStore,
    encode_inlined_item: encode_inlined_item,
    type_abbrevs: abbrev_map
};

fn reachable(ecx: @encode_ctxt, id: node_id) -> bool {
    ecx.reachable.contains_key(id)
}

fn encode_name(ecx: @encode_ctxt, ebml_w: ebml::Serializer, name: ident) {
    ebml_w.wr_tagged_str(tag_paths_data_name, ecx.tcx.sess.str_of(name));
}

fn encode_def_id(ebml_w: ebml::Serializer, id: def_id) {
    ebml_w.wr_tagged_str(tag_def_id, def_to_str(id));
}

fn encode_region_param(ecx: @encode_ctxt, ebml_w: ebml::Serializer,
                       it: @ast::item) {
    let opt_rp = ecx.tcx.region_paramd_items.find(it.id);
    for opt_rp.each |rp| {
        do ebml_w.wr_tag(tag_region_param) {
            (*rp).serialize(&ebml_w);
        }
    }
}

fn encode_mutability(ebml_w: ebml::Serializer, mt: class_mutability) {
    do ebml_w.wr_tag(tag_class_mut) {
        let val = match mt {
          class_immutable => 'a',
          class_mutable => 'm'
        };
        ebml_w.writer.write(&[val as u8]);
    }
}

type entry<T> = {val: T, pos: uint};

fn add_to_index(ecx: @encode_ctxt, ebml_w: ebml::Serializer, path: &[ident],
                index: &mut ~[entry<~str>], name: ident) {
    let mut full_path = ~[];
    full_path.push_all(path);
    full_path.push(name);
    index.push(
        {val: ast_util::path_name_i(full_path,
                                    ecx.tcx.sess.parse_sess.interner),
         pos: ebml_w.writer.tell()});
}

fn encode_trait_ref(ebml_w: ebml::Serializer, ecx: @encode_ctxt,
                    t: @trait_ref) {
    ebml_w.start_tag(tag_impl_trait);
    encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, t.ref_id));
    ebml_w.end_tag();
}


// Item info table encoding
fn encode_family(ebml_w: ebml::Serializer, c: char) {
    ebml_w.start_tag(tag_items_data_item_family);
    ebml_w.writer.write(&[c as u8]);
    ebml_w.end_tag();
}

fn def_to_str(did: def_id) -> ~str { fmt!("%d:%d", did.crate, did.node) }

fn encode_ty_type_param_bounds(ebml_w: ebml::Serializer, ecx: @encode_ctxt,
                               params: @~[ty::param_bounds]) {
    let ty_str_ctxt = @{diag: ecx.diag,
                        ds: def_to_str,
                        tcx: ecx.tcx,
                        reachable: |a| reachable(ecx, a),
                        abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)};
    for params.each |param| {
        ebml_w.start_tag(tag_items_data_item_ty_param_bounds);
        tyencode::enc_bounds(ebml_w.writer, ty_str_ctxt, *param);
        ebml_w.end_tag();
    }
}

fn encode_type_param_bounds(ebml_w: ebml::Serializer, ecx: @encode_ctxt,
                            params: ~[ty_param]) {
    let ty_param_bounds =
        @params.map(|param| ecx.tcx.ty_param_bounds.get(param.id));
    encode_ty_type_param_bounds(ebml_w, ecx, ty_param_bounds);
}


fn encode_variant_id(ebml_w: ebml::Serializer, vid: def_id) {
    ebml_w.start_tag(tag_items_data_item_variant);
    ebml_w.writer.write(str::to_bytes(def_to_str(vid)));
    ebml_w.end_tag();
}

fn write_type(ecx: @encode_ctxt, ebml_w: ebml::Serializer, typ: ty::t) {
    let ty_str_ctxt =
        @{diag: ecx.diag,
          ds: def_to_str,
          tcx: ecx.tcx,
          reachable: |a| reachable(ecx, a),
          abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)};
    tyencode::enc_ty(ebml_w.writer, ty_str_ctxt, typ);
}

fn write_vstore(ecx: @encode_ctxt, ebml_w: ebml::Serializer,
                vstore: ty::vstore) {
    let ty_str_ctxt =
        @{diag: ecx.diag,
          ds: def_to_str,
          tcx: ecx.tcx,
          reachable: |a| reachable(ecx, a),
          abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)};
    tyencode::enc_vstore(ebml_w.writer, ty_str_ctxt, vstore);
}

fn encode_type(ecx: @encode_ctxt, ebml_w: ebml::Serializer, typ: ty::t) {
    ebml_w.start_tag(tag_items_data_item_type);
    write_type(ecx, ebml_w, typ);
    ebml_w.end_tag();
}

fn encode_symbol(ecx: @encode_ctxt, ebml_w: ebml::Serializer, id: node_id) {
    ebml_w.start_tag(tag_items_data_item_symbol);
    let sym = match ecx.item_symbols.find(id) {
      Some(x) => x,
      None => {
        ecx.diag.handler().bug(
            fmt!("encode_symbol: id not found %d", id));
      }
    };
    ebml_w.writer.write(str::to_bytes(sym));
    ebml_w.end_tag();
}

fn encode_discriminant(ecx: @encode_ctxt, ebml_w: ebml::Serializer,
                       id: node_id) {
    ebml_w.start_tag(tag_items_data_item_symbol);
    ebml_w.writer.write(str::to_bytes(ecx.discrim_symbols.get(id)));
    ebml_w.end_tag();
}

fn encode_disr_val(_ecx: @encode_ctxt, ebml_w: ebml::Serializer,
                   disr_val: int) {
    ebml_w.start_tag(tag_disr_val);
    ebml_w.writer.write(str::to_bytes(int::to_str(disr_val,10u)));
    ebml_w.end_tag();
}

fn encode_parent_item(ebml_w: ebml::Serializer, id: def_id) {
    ebml_w.start_tag(tag_items_data_parent_item);
    ebml_w.writer.write(str::to_bytes(def_to_str(id)));
    ebml_w.end_tag();
}

fn encode_enum_variant_info(ecx: @encode_ctxt, ebml_w: ebml::Serializer,
                            id: node_id, variants: ~[variant],
                            path: ast_map::path, index: @mut ~[entry<int>],
                            ty_params: ~[ty_param]) {
    let mut disr_val = 0;
    let mut i = 0;
    let vi = ty::enum_variants(ecx.tcx, {crate: local_crate, node: id});
    for variants.each |variant| {
        index.push({val: variant.node.id, pos: ebml_w.writer.tell()});
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(variant.node.id));
        encode_family(ebml_w, 'v');
        encode_name(ecx, ebml_w, variant.node.name);
        encode_parent_item(ebml_w, local_def(id));
        encode_type(ecx, ebml_w,
                    node_id_to_type(ecx.tcx, variant.node.id));
        match variant.node.kind {
            ast::tuple_variant_kind(args)
                    if args.len() > 0 && ty_params.len() == 0 => {
                encode_symbol(ecx, ebml_w, variant.node.id);
            }
            ast::tuple_variant_kind(_) | ast::struct_variant_kind(_) |
            ast::enum_variant_kind(_) => {}
        }
        encode_discriminant(ecx, ebml_w, variant.node.id);
        if vi[i].disr_val != disr_val {
            encode_disr_val(ecx, ebml_w, vi[i].disr_val);
            disr_val = vi[i].disr_val;
        }
        encode_type_param_bounds(ebml_w, ecx, ty_params);
        encode_path(ecx, ebml_w, path, ast_map::path_name(variant.node.name));
        ebml_w.end_tag();
        disr_val += 1;
        i += 1;
    }
}

fn encode_path(ecx: @encode_ctxt, ebml_w: ebml::Serializer,
               path: ast_map::path, name: ast_map::path_elt) {
    fn encode_path_elt(ecx: @encode_ctxt, ebml_w: ebml::Serializer,
                       elt: ast_map::path_elt) {
        let (tag, name) = match elt {
          ast_map::path_mod(name) => (tag_path_elt_mod, name),
          ast_map::path_name(name) => (tag_path_elt_name, name)
        };

        ebml_w.wr_tagged_str(tag, ecx.tcx.sess.str_of(name));
    }

    do ebml_w.wr_tag(tag_path) {
        ebml_w.wr_tagged_u32(tag_path_len, (vec::len(path) + 1u) as u32);
        for vec::each(path) |pe| {
            encode_path_elt(ecx, ebml_w, *pe);
        }
        encode_path_elt(ecx, ebml_w, name);
    }
}

fn encode_info_for_mod(ecx: @encode_ctxt, ebml_w: ebml::Serializer, md: _mod,
                       id: node_id, path: ast_map::path, name: ident) {
    ebml_w.start_tag(tag_items_data_item);
    encode_def_id(ebml_w, local_def(id));
    encode_family(ebml_w, 'm');
    encode_name(ecx, ebml_w, name);
    debug!("(encoding info for module) encoding info for module ID %d", id);

    // Encode info about all the module children.
    for md.items.each |item| {
        match item.node {
            item_impl(*) | item_class(*) => {
                let (ident, did) = (item.ident, item.id);
                debug!("(encoding info for module) ... encoding impl %s \
                        (%?/%?), exported? %?",
                        ecx.tcx.sess.str_of(ident),
                        did,
                        ast_map::node_id_to_str(ecx.tcx.items, did, ecx.tcx
                                                .sess.parse_sess.interner),
                        ast_util::is_exported(ident, md));

                ebml_w.start_tag(tag_mod_impl);
                ebml_w.wr_str(def_to_str(local_def(did)));
                ebml_w.end_tag();
            }
            _ => {} // XXX: Encode these too.
        }
    }

    encode_path(ecx, ebml_w, path, ast_map::path_mod(name));

    // Encode the reexports of this module.
    debug!("(encoding info for module) encoding reexports for %d", id);
    match ecx.reexports2.find(id) {
        Some(exports) => {
            debug!("(encoding info for module) found reexports for %d", id);
            for exports.each |exp| {
                debug!("(encoding info for module) reexport '%s' for %d",
                       exp.name, id);
                ebml_w.start_tag(tag_items_data_item_reexport);
                ebml_w.start_tag(tag_items_data_item_reexport_def_id);
                ebml_w.wr_str(def_to_str(exp.def_id));
                ebml_w.end_tag();
                ebml_w.start_tag(tag_items_data_item_reexport_name);
                ebml_w.wr_str(exp.name);
                ebml_w.end_tag();
                ebml_w.end_tag();
            }
        }
        None => {
            debug!("(encoding info for module) found no reexports for %d",
                   id);
        }
    }

    ebml_w.end_tag();
}

fn encode_visibility(ebml_w: ebml::Serializer, visibility: visibility) {
    encode_family(ebml_w, match visibility {
        public => 'g',
        private => 'j',
        inherited => 'N'
    });
}

fn encode_self_type(ebml_w: ebml::Serializer, self_type: ast::self_ty_) {
    ebml_w.start_tag(tag_item_trait_method_self_ty);

    // Encode the base self type.
    let ch;
    match self_type {
        sty_static =>       { ch = 's' as u8; }
        sty_by_ref =>       { ch = 'r' as u8; }
        sty_value =>        { ch = 'v' as u8; }
        sty_region(_) =>    { ch = '&' as u8; }
        sty_box(_) =>       { ch = '@' as u8; }
        sty_uniq(_) =>      { ch = '~' as u8; }
    }
    ebml_w.writer.write(&[ ch ]);

    // Encode mutability.
    match self_type {
        sty_static | sty_by_ref | sty_value => { /* No-op. */ }
        sty_region(m_imm) | sty_box(m_imm) | sty_uniq(m_imm) => {
            ebml_w.writer.write(&[ 'i' as u8 ]);
        }
        sty_region(m_mutbl) | sty_box(m_mutbl) | sty_uniq(m_mutbl) => {
            ebml_w.writer.write(&[ 'm' as u8 ]);
        }
        sty_region(m_const) | sty_box(m_const) | sty_uniq(m_const) => {
            ebml_w.writer.write(&[ 'c' as u8 ]);
        }
    }

    ebml_w.end_tag();
}

fn encode_method_sort(ebml_w: ebml::Serializer, sort: char) {
    ebml_w.start_tag(tag_item_trait_method_sort);
    ebml_w.writer.write(&[ sort as u8 ]);
    ebml_w.end_tag();
}

/* Returns an index of items in this class */
fn encode_info_for_class(ecx: @encode_ctxt, ebml_w: ebml::Serializer,
                         id: node_id, path: ast_map::path,
                         class_tps: ~[ty_param],
                         fields: ~[@struct_field],
                         methods: ~[@method],
                         global_index: @mut~[entry<int>]) -> ~[entry<int>] {
    /* Each class has its own index, since different classes
       may have fields with the same name */
    let index = @mut ~[];
    let tcx = ecx.tcx;
     /* We encode both private and public fields -- need to include
        private fields to get the offsets right */
    for fields.each |field| {
        match field.node.kind {
            named_field(nm, mt, vis) => {
                let id = field.node.id;
                index.push({val: id, pos: ebml_w.writer.tell()});
                global_index.push({val: id,
                                    pos: ebml_w.writer.tell()});
                ebml_w.start_tag(tag_items_data_item);
                debug!("encode_info_for_class: doing %s %d",
                       tcx.sess.str_of(nm), id);
                encode_visibility(ebml_w, vis);
                encode_name(ecx, ebml_w, nm);
                encode_path(ecx, ebml_w, path, ast_map::path_name(nm));
                encode_type(ecx, ebml_w, node_id_to_type(tcx, id));
                encode_mutability(ebml_w, mt);
                encode_def_id(ebml_w, local_def(id));
                ebml_w.end_tag();
            }
            unnamed_field => {}
        }
    }

    for methods.each |m| {
        match m.vis {
            public | inherited => {
                index.push({val: m.id, pos: ebml_w.writer.tell()});
                global_index.push(
                    {val: m.id, pos: ebml_w.writer.tell()});
                let impl_path = vec::append_one(path,
                                                ast_map::path_name(m.ident));
                debug!("encode_info_for_class: doing %s %d",
                       ecx.tcx.sess.str_of(m.ident), m.id);
                encode_info_for_method(ecx, ebml_w, impl_path,
                                       should_inline(m.attrs), id, *m,
                                       vec::append(class_tps, m.tps));
            }
            _ => { /* don't encode private methods */ }
        }
    }

    *index
}

// This is for encoding info for ctors and dtors
fn encode_info_for_ctor(ecx: @encode_ctxt, ebml_w: ebml::Serializer,
                        id: node_id, ident: ident, path: ast_map::path,
                        item: Option<inlined_item>, tps: ~[ty_param]) {
        ebml_w.start_tag(tag_items_data_item);
        encode_name(ecx, ebml_w, ident);
        encode_def_id(ebml_w, local_def(id));
        encode_family(ebml_w, purity_fn_family(ast::impure_fn));
        encode_type_param_bounds(ebml_w, ecx, tps);
        let its_ty = node_id_to_type(ecx.tcx, id);
        debug!("fn name = %s ty = %s its node id = %d",
               ecx.tcx.sess.str_of(ident),
               util::ppaux::ty_to_str(ecx.tcx, its_ty), id);
        encode_type(ecx, ebml_w, its_ty);
        encode_path(ecx, ebml_w, path, ast_map::path_name(ident));
        match item {
           Some(it) => {
             ecx.encode_inlined_item(ecx, ebml_w, path, it);
           }
           None => {
             encode_symbol(ecx, ebml_w, id);
           }
        }
        ebml_w.end_tag();
}

fn encode_info_for_method(ecx: @encode_ctxt, ebml_w: ebml::Serializer,
                          impl_path: ast_map::path, should_inline: bool,
                          parent_id: node_id,
                          m: @method, all_tps: ~[ty_param]) {
    debug!("encode_info_for_method: %d %s %u", m.id,
           ecx.tcx.sess.str_of(m.ident), all_tps.len());
    ebml_w.start_tag(tag_items_data_item);
    encode_def_id(ebml_w, local_def(m.id));
    encode_family(ebml_w, purity_fn_family(m.purity));
    encode_type_param_bounds(ebml_w, ecx, all_tps);
    encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, m.id));
    encode_name(ecx, ebml_w, m.ident);
    encode_path(ecx, ebml_w, impl_path, ast_map::path_name(m.ident));
    encode_self_type(ebml_w, m.self_ty.node);
    if all_tps.len() > 0u || should_inline {
        ecx.encode_inlined_item(
           ecx, ebml_w, impl_path,
           ii_method(local_def(parent_id), m));
    } else {
        encode_symbol(ecx, ebml_w, m.id);
    }
    ebml_w.end_tag();
}

fn purity_fn_family(p: purity) -> char {
    match p {
      unsafe_fn => 'u',
      pure_fn => 'p',
      impure_fn => 'f',
      extern_fn => 'e'
    }
}
fn purity_static_method_family(p: purity) -> char {
    match p {
      unsafe_fn => 'U',
      pure_fn => 'P',
      impure_fn => 'F',
      _ => fail ~"extern fn can't be static"
    }
}


fn should_inline(attrs: ~[attribute]) -> bool {
    match attr::find_inline_attr(attrs) {
        attr::ia_none | attr::ia_never  => false,
        attr::ia_hint | attr::ia_always => true
    }
}


fn encode_info_for_item(ecx: @encode_ctxt, ebml_w: ebml::Serializer,
                        item: @item, index: @mut ~[entry<int>],
                        path: ast_map::path) {

    let tcx = ecx.tcx;
    let must_write =
        match item.node {
          item_enum(_, _) | item_impl(*)
          | item_trait(*) | item_class(*) => true,
          _ => false
        };
    if !must_write && !reachable(ecx, item.id) { return; }

    fn add_to_index_(item: @item, ebml_w: ebml::Serializer,
                     index: @mut ~[entry<int>]) {
        index.push({val: item.id, pos: ebml_w.writer.tell()});
    }
    let add_to_index = |copy ebml_w| add_to_index_(item, ebml_w, index);

    match item.node {
      item_const(_, _) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'c');
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_symbol(ecx, ebml_w, item.id);
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        ebml_w.end_tag();
      }
      item_fn(_, purity, tps, _) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, purity_fn_family(purity));
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        encode_attributes(ebml_w, item.attrs);
        if tps.len() > 0u || should_inline(item.attrs) {
            ecx.encode_inlined_item(ecx, ebml_w, path, ii_item(item));
        } else {
            encode_symbol(ecx, ebml_w, item.id);
        }
        ebml_w.end_tag();
      }
      item_mod(m) => {
        add_to_index();
        encode_info_for_mod(ecx, ebml_w, m, item.id, path, item.ident);
      }
      item_foreign_mod(_) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'n');
        encode_name(ecx, ebml_w, item.ident);
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        ebml_w.end_tag();
      }
      item_ty(_, tps) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'y');
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_name(ecx, ebml_w, item.ident);
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        encode_region_param(ecx, ebml_w, item);
        ebml_w.end_tag();
      }
      item_enum(enum_definition, tps) => {
        add_to_index();
        do ebml_w.wr_tag(tag_items_data_item) {
            encode_def_id(ebml_w, local_def(item.id));
            encode_family(ebml_w, 't');
            encode_type_param_bounds(ebml_w, ecx, tps);
            encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
            encode_name(ecx, ebml_w, item.ident);
            for enum_definition.variants.each |v| {
                encode_variant_id(ebml_w, local_def(v.node.id));
            }
            ecx.encode_inlined_item(ecx, ebml_w, path, ii_item(item));
            encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
            encode_region_param(ecx, ebml_w, item);
        }
        encode_enum_variant_info(ecx, ebml_w, item.id,
                                 enum_definition.variants, path, index, tps);
      }
      item_class(struct_def, tps) => {
        /* First, encode the fields and methods
           These come first because we need to write them to make
           the index, and the index needs to be in the item for the
           class itself */
        let idx = encode_info_for_class(ecx, ebml_w, item.id, path, tps,
                                        struct_def.fields, struct_def.methods,
                                        index);
        /* Encode the dtor */
        do struct_def.dtor.iter |dtor| {
            index.push({val: dtor.node.id, pos: ebml_w.writer.tell()});
          encode_info_for_ctor(ecx, ebml_w, dtor.node.id,
                               ecx.tcx.sess.ident_of(
                                   ecx.tcx.sess.str_of(item.ident) +
                                   ~"_dtor"),
                               path, if tps.len() > 0u {
                                   Some(ii_dtor(*dtor, item.ident, tps,
                                                local_def(item.id))) }
                               else { None }, tps);
        }

        /* Index the class*/
        add_to_index();
        /* Now, make an item for the class itself */
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'S');
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_name(ecx, ebml_w, item.ident);
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        encode_region_param(ecx, ebml_w, item);
        for struct_def.traits.each |t| {
           encode_trait_ref(ebml_w, ecx, *t);
        }
        /* Encode the dtor */
        /* Encode id for dtor */
        do struct_def.dtor.iter |dtor| {
            do ebml_w.wr_tag(tag_item_dtor) {
                encode_def_id(ebml_w, local_def(dtor.node.id));
            }
        };

        /* Encode def_ids for each field and method
         for methods, write all the stuff get_trait_method
        needs to know*/
        for struct_def.fields.each |f| {
            match f.node.kind {
                named_field(ident, _, vis) => {
                   ebml_w.start_tag(tag_item_field);
                   encode_visibility(ebml_w, vis);
                   encode_name(ecx, ebml_w, ident);
                   encode_def_id(ebml_w, local_def(f.node.id));
                   ebml_w.end_tag();
                }
                unnamed_field => {}
            }
        }

        for struct_def.methods.each |m| {
           match m.vis {
              private => { /* do nothing */ }
              public | inherited => {
                /* Write the info that's needed when viewing this class
                   as a trait */
                ebml_w.start_tag(tag_item_trait_method);
                encode_family(ebml_w, purity_fn_family(m.purity));
                encode_name(ecx, ebml_w, m.ident);
                encode_type_param_bounds(ebml_w, ecx, m.tps);
                encode_type(ecx, ebml_w, node_id_to_type(tcx, m.id));
                encode_def_id(ebml_w, local_def(m.id));
                encode_self_type(ebml_w, m.self_ty.node);
                ebml_w.end_tag();
                /* Write the info that's needed when viewing this class
                   as an impl (just the method def_id and self type) */
                ebml_w.start_tag(tag_item_impl_method);
                ebml_w.writer.write(to_bytes(def_to_str(local_def(m.id))));
                ebml_w.end_tag();
              }
           }
        }
        /* Each class has its own index -- encode it */
        let bkts = create_index(idx);
        encode_index(ebml_w, bkts, write_int);
        ebml_w.end_tag();
      }
      item_impl(tps, opt_trait, _, methods) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'i');
        encode_region_param(ecx, ebml_w, item);
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_name(ecx, ebml_w, item.ident);
        encode_attributes(ebml_w, item.attrs);
        for methods.each |m| {
            ebml_w.start_tag(tag_item_impl_method);
            ebml_w.writer.write(str::to_bytes(def_to_str(local_def(m.id))));
            ebml_w.end_tag();
        }
        do opt_trait.iter() |associated_trait| {
           encode_trait_ref(ebml_w, ecx, *associated_trait)
        }
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        ebml_w.end_tag();

        let impl_path = vec::append_one(path,
                                        ast_map::path_name(item.ident));
        for methods.each |m| {
            index.push({val: m.id, pos: ebml_w.writer.tell()});
            encode_info_for_method(ecx, ebml_w, impl_path,
                                   should_inline(m.attrs), item.id, *m,
                                   vec::append(tps, m.tps));
        }
      }
      item_trait(tps, traits, ms) => {
        let provided_methods = dvec::DVec();

        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'I');
        encode_region_param(ecx, ebml_w, item);
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_name(ecx, ebml_w, item.ident);
        encode_attributes(ebml_w, item.attrs);
        let mut i = 0u;
        for vec::each(*ty::trait_methods(tcx, local_def(item.id))) |mty| {
            match ms[i] {
              required(ty_m) => {
                ebml_w.start_tag(tag_item_trait_method);
                encode_def_id(ebml_w, local_def(ty_m.id));
                encode_name(ecx, ebml_w, mty.ident);
                encode_type_param_bounds(ebml_w, ecx, ty_m.tps);
                encode_type(ecx, ebml_w, ty::mk_fn(tcx, mty.fty));
                encode_family(ebml_w, purity_fn_family(mty.fty.meta.purity));
                encode_self_type(ebml_w, mty.self_ty);
                encode_method_sort(ebml_w, 'r');
                ebml_w.end_tag();
              }
              provided(m) => {
                provided_methods.push(m);

                ebml_w.start_tag(tag_item_trait_method);
                encode_def_id(ebml_w, local_def(m.id));
                encode_name(ecx, ebml_w, mty.ident);
                encode_type_param_bounds(ebml_w, ecx, m.tps);
                encode_type(ecx, ebml_w, ty::mk_fn(tcx, mty.fty));
                encode_family(ebml_w, purity_fn_family(mty.fty.meta.purity));
                encode_self_type(ebml_w, mty.self_ty);
                encode_method_sort(ebml_w, 'p');
                ebml_w.end_tag();
              }
            }
            i += 1u;
        }
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        for traits.each |associated_trait| {
           encode_trait_ref(ebml_w, ecx, *associated_trait)
        }
        ebml_w.end_tag();

        // Now, output all of the static methods as items.  Note that for the
        // method info, we output static methods with type signatures as
        // written. Here, we output the *real* type signatures. I feel like
        // maybe we should only ever handle the real type signatures.
        for vec::each(ms) |m| {
            let ty_m = ast_util::trait_method_to_ty_method(*m);
            if ty_m.self_ty.node != ast::sty_static { loop; }

            index.push({val: ty_m.id, pos: ebml_w.writer.tell()});

            ebml_w.start_tag(tag_items_data_item);
            encode_def_id(ebml_w, local_def(ty_m.id));
            encode_parent_item(ebml_w, local_def(item.id));
            encode_name(ecx, ebml_w, ty_m.ident);
            encode_family(ebml_w,
                          purity_static_method_family(ty_m.purity));
            let polyty = ecx.tcx.tcache.get(local_def(ty_m.id));
            encode_ty_type_param_bounds(ebml_w, ecx, polyty.bounds);
            encode_type(ecx, ebml_w, polyty.ty);
            encode_path(ecx, ebml_w, path, ast_map::path_name(ty_m.ident));
            ebml_w.end_tag();
        }

        // Finally, output all the provided methods as items.
        for provided_methods.each |m| {
            index.push({val: m.id, pos: ebml_w.writer.tell()});
            encode_info_for_method(ecx, ebml_w, path, true, item.id, *m,
                                   m.tps);
        }
      }
      item_mac(*) => fail ~"item macros unimplemented"
    }
}

fn encode_info_for_foreign_item(ecx: @encode_ctxt, ebml_w: ebml::Serializer,
                                nitem: @foreign_item,
                                index: @mut ~[entry<int>],
                                path: ast_map::path, abi: foreign_abi) {
    if !reachable(ecx, nitem.id) { return; }
    index.push({val: nitem.id, pos: ebml_w.writer.tell()});

    ebml_w.start_tag(tag_items_data_item);
    match nitem.node {
      foreign_item_fn(_, purity, tps) => {
        encode_def_id(ebml_w, local_def(nitem.id));
        encode_family(ebml_w, purity_fn_family(purity));
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, nitem.id));
        if abi == foreign_abi_rust_intrinsic {
            ecx.encode_inlined_item(ecx, ebml_w, path,
                                    ii_foreign(nitem));
        } else {
            encode_symbol(ecx, ebml_w, nitem.id);
        }
        encode_path(ecx, ebml_w, path, ast_map::path_name(nitem.ident));
      }
      foreign_item_const(*) => {
        encode_def_id(ebml_w, local_def(nitem.id));
        encode_family(ebml_w, 'c');
        encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, nitem.id));
        encode_symbol(ecx, ebml_w, nitem.id);
        encode_path(ecx, ebml_w, path, ast_map::path_name(nitem.ident));
      }
    }
    ebml_w.end_tag();
}

fn encode_info_for_items(ecx: @encode_ctxt, ebml_w: ebml::Serializer,
                         crate: @crate) -> ~[entry<int>] {
    let index = @mut ~[];
    ebml_w.start_tag(tag_items_data);
    index.push({val: crate_node_id, pos: ebml_w.writer.tell()});
    encode_info_for_mod(ecx, ebml_w, crate.node.module,
                        crate_node_id, ~[],
                        syntax::parse::token::special_idents::invalid);
    visit::visit_crate(*crate, (), visit::mk_vt(@{
        visit_expr: |_e, _cx, _v| { },
        visit_item: |i, cx, v, copy ebml_w| {
            visit::visit_item(i, cx, v);
            match ecx.tcx.items.get(i.id) {
              ast_map::node_item(_, pt) => {
                encode_info_for_item(ecx, ebml_w, i, index, *pt);
              }
              _ => fail ~"bad item"
            }
        },
        visit_foreign_item: |ni, cx, v, copy ebml_w| {
            visit::visit_foreign_item(ni, cx, v);
            match ecx.tcx.items.get(ni.id) {
              ast_map::node_foreign_item(_, abi, pt) => {
                encode_info_for_foreign_item(ecx, ebml_w, ni,
                                             index, *pt, abi);
              }
              // case for separate item and foreign-item tables
              _ => fail ~"bad foreign item"
            }
        }
        ,.. *visit::default_visitor()
    }));
    ebml_w.end_tag();
    return *index;
}


// Path and definition ID indexing

fn create_index<T: Copy Hash IterBytes>(index: ~[entry<T>]) ->
   ~[@~[entry<T>]] {
    let mut buckets: ~[@mut ~[entry<T>]] = ~[];
    for uint::range(0u, 256u) |_i| { buckets.push(@mut ~[]); };
    for index.each |elt| {
        let h = elt.val.hash() as uint;
        buckets[h % 256].push(*elt);
    }

    let mut buckets_frozen = ~[];
    for buckets.each |bucket| {
        buckets_frozen.push(@**bucket);
    }
    return buckets_frozen;
}

fn encode_index<T>(ebml_w: ebml::Serializer, buckets: ~[@~[entry<T>]],
                   write_fn: fn(io::Writer, T)) {
    let writer = ebml_w.writer;
    ebml_w.start_tag(tag_index);
    let mut bucket_locs: ~[uint] = ~[];
    ebml_w.start_tag(tag_index_buckets);
    for buckets.each |bucket| {
        bucket_locs.push(ebml_w.writer.tell());
        ebml_w.start_tag(tag_index_buckets_bucket);
        for vec::each(**bucket) |elt| {
            ebml_w.start_tag(tag_index_buckets_bucket_elt);
            assert elt.pos < 0xffff_ffff;
            writer.write_be_u32(elt.pos as u32);
            write_fn(writer, elt.val);
            ebml_w.end_tag();
        }
        ebml_w.end_tag();
    }
    ebml_w.end_tag();
    ebml_w.start_tag(tag_index_table);
    for bucket_locs.each |pos| {
        assert *pos < 0xffff_ffff;
        writer.write_be_u32(*pos as u32);
    }
    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn write_str(writer: io::Writer, &&s: ~str) { writer.write_str(s); }

fn write_int(writer: io::Writer, &&n: int) {
    assert n < 0x7fff_ffff;
    writer.write_be_u32(n as u32);
}

fn encode_meta_item(ebml_w: ebml::Serializer, mi: meta_item) {
    match mi.node {
      meta_word(name) => {
        ebml_w.start_tag(tag_meta_item_word);
        ebml_w.start_tag(tag_meta_item_name);
        ebml_w.writer.write(str::to_bytes(name));
        ebml_w.end_tag();
        ebml_w.end_tag();
      }
      meta_name_value(name, value) => {
        match value.node {
          lit_str(value) => {
            ebml_w.start_tag(tag_meta_item_name_value);
            ebml_w.start_tag(tag_meta_item_name);
            ebml_w.writer.write(str::to_bytes(name));
            ebml_w.end_tag();
            ebml_w.start_tag(tag_meta_item_value);
            ebml_w.writer.write(str::to_bytes(*value));
            ebml_w.end_tag();
            ebml_w.end_tag();
          }
          _ => {/* FIXME (#623): encode other variants */ }
        }
      }
      meta_list(name, items) => {
        ebml_w.start_tag(tag_meta_item_list);
        ebml_w.start_tag(tag_meta_item_name);
        ebml_w.writer.write(str::to_bytes(name));
        ebml_w.end_tag();
        for items.each |inner_item| {
            encode_meta_item(ebml_w, **inner_item);
        }
        ebml_w.end_tag();
      }
    }
}

fn encode_attributes(ebml_w: ebml::Serializer, attrs: ~[attribute]) {
    ebml_w.start_tag(tag_attributes);
    for attrs.each |attr| {
        ebml_w.start_tag(tag_attribute);
        encode_meta_item(ebml_w, attr.node.value);
        ebml_w.end_tag();
    }
    ebml_w.end_tag();
}

// So there's a special crate attribute called 'link' which defines the
// metadata that Rust cares about for linking crates. This attribute requires
// 'name' and 'vers' items, so if the user didn't provide them we will throw
// them in anyway with default values.
fn synthesize_crate_attrs(ecx: @encode_ctxt, crate: @crate) -> ~[attribute] {

    fn synthesize_link_attr(ecx: @encode_ctxt, items: ~[@meta_item]) ->
       attribute {

        assert (ecx.link_meta.name != ~"");
        assert (ecx.link_meta.vers != ~"");

        let name_item =
            attr::mk_name_value_item_str(~"name", ecx.link_meta.name);
        let vers_item =
            attr::mk_name_value_item_str(~"vers", ecx.link_meta.vers);

        let other_items =
            {
                let tmp = attr::remove_meta_items_by_name(items, ~"name");
                attr::remove_meta_items_by_name(tmp, ~"vers")
            };

        let meta_items = vec::append(~[name_item, vers_item], other_items);
        let link_item = attr::mk_list_item(~"link", meta_items);

        return attr::mk_attr(link_item);
    }

    let mut attrs: ~[attribute] = ~[];
    let mut found_link_attr = false;
    for crate.node.attrs.each |attr| {
        attrs.push(
            if attr::get_attr_name(*attr) != ~"link" {
                *attr
            } else {
                match attr.node.value.node {
                  meta_list(_, l) => {
                    found_link_attr = true;;
                    synthesize_link_attr(ecx, l)
                  }
                  _ => *attr
                }
            });
    }

    if !found_link_attr { attrs.push(synthesize_link_attr(ecx, ~[])); }

    return attrs;
}

fn encode_crate_deps(ecx: @encode_ctxt, ebml_w: ebml::Serializer,
                     cstore: cstore::CStore) {

    fn get_ordered_deps(ecx: @encode_ctxt, cstore: cstore::CStore)
        -> ~[decoder::crate_dep] {

        type hashkv = @{key: crate_num, val: cstore::crate_metadata};
        type numdep = decoder::crate_dep;

        // Pull the cnums and name,vers,hash out of cstore
        let mut deps: ~[numdep] = ~[];
        do cstore::iter_crate_data(cstore) |key, val| {
            let dep = {cnum: key, name: ecx.tcx.sess.ident_of(val.name),
                       vers: decoder::get_crate_vers(val.data),
                       hash: decoder::get_crate_hash(val.data)};
            deps.push(dep);
        };

        // Sort by cnum
        pure fn lteq(kv1: &numdep, kv2: &numdep) -> bool {
            kv1.cnum <= kv2.cnum
        }
        std::sort::quick_sort(lteq, deps);

        // Sanity-check the crate numbers
        let mut expected_cnum = 1;
        for deps.each |n| {
            assert (n.cnum == expected_cnum);
            expected_cnum += 1;
        }

        // mut -> immutable hack for vec::map
        return vec::slice(deps, 0u, vec::len(deps));
    }

    // We're just going to write a list of crate 'name-hash-version's, with
    // the assumption that they are numbered 1 to n.
    // FIXME (#2166): This is not nearly enough to support correct versioning
    // but is enough to get transitive crate dependencies working.
    ebml_w.start_tag(tag_crate_deps);
    for get_ordered_deps(ecx, cstore).each |dep| {
        encode_crate_dep(ecx, ebml_w, *dep);
    }
    ebml_w.end_tag();
}

fn encode_crate_dep(ecx: @encode_ctxt, ebml_w: ebml::Serializer,
                    dep: decoder::crate_dep) {
    ebml_w.start_tag(tag_crate_dep);
    ebml_w.start_tag(tag_crate_dep_name);
    ebml_w.writer.write(str::to_bytes(ecx.tcx.sess.str_of(dep.name)));
    ebml_w.end_tag();
    ebml_w.start_tag(tag_crate_dep_vers);
    ebml_w.writer.write(str::to_bytes(dep.vers));
    ebml_w.end_tag();
    ebml_w.start_tag(tag_crate_dep_hash);
    ebml_w.writer.write(str::to_bytes(dep.hash));
    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn encode_hash(ebml_w: ebml::Serializer, hash: ~str) {
    ebml_w.start_tag(tag_crate_hash);
    ebml_w.writer.write(str::to_bytes(hash));
    ebml_w.end_tag();
}

// NB: Increment this as you change the metadata encoding version.
const metadata_encoding_version : &[u8] = &[0x72, //'r' as u8,
                                            0x75, //'u' as u8,
                                            0x73, //'s' as u8,
                                            0x74, //'t' as u8,
                                            0, 0, 0, 1 ];

fn encode_metadata(parms: encode_parms, crate: @crate) -> ~[u8] {
    let wr = @io::BytesWriter();
    let stats =
        {mut inline_bytes: 0,
         mut attr_bytes: 0,
         mut dep_bytes: 0,
         mut item_bytes: 0,
         mut index_bytes: 0,
         mut zero_bytes: 0,
         mut total_bytes: 0,
         mut n_inlines: 0};
    let ecx: @encode_ctxt = @encode_ctxt({
        diag: parms.diag,
        tcx: parms.tcx,
        stats: move stats,
        reachable: parms.reachable,
        reexports2: parms.reexports2,
        item_symbols: parms.item_symbols,
        discrim_symbols: parms.discrim_symbols,
        link_meta: parms.link_meta,
        cstore: parms.cstore,
        encode_inlined_item: parms.encode_inlined_item,
        type_abbrevs: ty::new_ty_hash()
     });

    let ebml_w = ebml::Serializer(wr as io::Writer);

    encode_hash(ebml_w, ecx.link_meta.extras_hash);

    let mut i = wr.pos;
    let crate_attrs = synthesize_crate_attrs(ecx, crate);
    encode_attributes(ebml_w, crate_attrs);
    ecx.stats.attr_bytes = wr.pos - i;

    i = wr.pos;
    encode_crate_deps(ecx, ebml_w, ecx.cstore);
    ecx.stats.dep_bytes = wr.pos - i;

    // Encode and index the items.
    ebml_w.start_tag(tag_items);
    i = wr.pos;
    let items_index = encode_info_for_items(ecx, ebml_w, crate);
    ecx.stats.item_bytes = wr.pos - i;

    i = wr.pos;
    let items_buckets = create_index(items_index);
    encode_index(ebml_w, items_buckets, write_int);
    ecx.stats.index_bytes = wr.pos - i;
    ebml_w.end_tag();

    ecx.stats.total_bytes = wr.pos;

    if (parms.tcx.sess.meta_stats()) {

        do wr.bytes.borrow |v| {
            do v.each |e| {
                if *e == 0 {
                    ecx.stats.zero_bytes += 1;
                }
                true
            }
        }

        io::println("metadata stats:");
        io::println(fmt!("    inline bytes: %u", ecx.stats.inline_bytes));
        io::println(fmt!(" attribute bytes: %u", ecx.stats.attr_bytes));
        io::println(fmt!("       dep bytes: %u", ecx.stats.dep_bytes));
        io::println(fmt!("      item bytes: %u", ecx.stats.item_bytes));
        io::println(fmt!("     index bytes: %u", ecx.stats.index_bytes));
        io::println(fmt!("      zero bytes: %u", ecx.stats.zero_bytes));
        io::println(fmt!("     total bytes: %u", ecx.stats.total_bytes));
    }

    // Pad this, since something (LLVM, presumably) is cutting off the
    // remaining % 4 bytes.
    wr.write(&[0u8, 0u8, 0u8, 0u8]);

    // FIXME #3396: weird bug here, for reasons unclear this emits random
    // looking bytes (mostly 0x1) if we use the version byte-array constant
    // above; so we use a string constant inline instead.
    //
    // Should be:
    //
    //   vec::from_slice(metadata_encoding_version) +

    (do str::as_bytes(&~"rust\x00\x00\x00\x01") |bytes| {
        vec::slice(*bytes, 0, 8)
    }) + flate::deflate_bytes(wr.bytes.check_out(|buf| buf))
}

// Get the encoded string for a type
fn encoded_ty(tcx: ty::ctxt, t: ty::t) -> ~str {
    let cx = @{diag: tcx.diag,
               ds: def_to_str,
               tcx: tcx,
               reachable: |_id| false,
               abbrevs: tyencode::ac_no_abbrevs};
    do io::with_str_writer |wr| {
        tyencode::enc_ty(wr, cx, t);
    }
}


// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
