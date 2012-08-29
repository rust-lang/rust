// Searching for information from the cstore

import std::{ebml};
import syntax::ast;
import syntax::ast_util;
import syntax::ast_map;
import middle::ty;
import option::{Some, None};
import syntax::diagnostic::span_handler;
import syntax::diagnostic::expect;
import ast_util::dummy_sp;
import common::*;
import std::map::hashmap;
import dvec::DVec;

export class_dtor;
export get_symbol;
export get_class_fields;
export get_class_method;
export get_field_type;
export get_type_param_count;
export get_region_param;
export get_enum_variants;
export get_impls_for_mod;
export get_trait_methods;
export get_method_names_if_trait;
export get_item_attrs;
export each_path;
export get_type;
export get_impl_traits;
export get_impl_method;
export get_item_path;
export maybe_get_item_ast, found_ast, found, found_parent, not_found;

fn get_symbol(cstore: cstore::cstore, def: ast::def_id) -> ~str {
    let cdata = cstore::get_crate_data(cstore, def.crate).data;
    return decoder::get_symbol(cdata, def.node);
}

fn get_type_param_count(cstore: cstore::cstore, def: ast::def_id) -> uint {
    let cdata = cstore::get_crate_data(cstore, def.crate).data;
    return decoder::get_type_param_count(cdata, def.node);
}

/// Iterates over all the paths in the given crate.
fn each_path(cstore: cstore::cstore, cnum: ast::crate_num,
             f: fn(decoder::path_entry) -> bool) {
    let crate_data = cstore::get_crate_data(cstore, cnum);
    decoder::each_path(cstore.intr, crate_data, f);
}

fn get_item_path(tcx: ty::ctxt, def: ast::def_id) -> ast_map::path {
    let cstore = tcx.cstore;
    let cdata = cstore::get_crate_data(cstore, def.crate);
    let path = decoder::get_item_path(cstore.intr, cdata, def.node);

    // FIXME #1920: This path is not always correct if the crate is not linked
    // into the root namespace.
    vec::append(~[ast_map::path_mod(tcx.sess.ident_of(cdata.name))], path)
}

enum found_ast {
    found(ast::inlined_item),
    found_parent(ast::def_id, ast::inlined_item),
    not_found,
}

// Finds the AST for this item in the crate metadata, if any.  If the item was
// not marked for inlining, then the AST will not be present and hence none
// will be returned.
fn maybe_get_item_ast(tcx: ty::ctxt, def: ast::def_id,
                      decode_inlined_item: decoder::decode_inlined_item)
    -> found_ast {
    let cstore = tcx.cstore;
    let cdata = cstore::get_crate_data(cstore, def.crate);
    decoder::maybe_get_item_ast(cstore.intr, cdata, tcx, def.node,
                                decode_inlined_item)
}

fn get_enum_variants(tcx: ty::ctxt, def: ast::def_id)
    -> ~[ty::variant_info] {
    let cstore = tcx.cstore;
    let cdata = cstore::get_crate_data(cstore, def.crate);
    return decoder::get_enum_variants(cstore.intr, cdata, def.node, tcx)
}

fn get_impls_for_mod(cstore: cstore::cstore, def: ast::def_id,
                     name: Option<ast::ident>)
    -> @~[@decoder::_impl] {
    let cdata = cstore::get_crate_data(cstore, def.crate);
    do decoder::get_impls_for_mod(cstore.intr, cdata, def.node, name) |cnum| {
        cstore::get_crate_data(cstore, cnum)
    }
}

fn get_trait_methods(tcx: ty::ctxt, def: ast::def_id) -> @~[ty::method] {
    let cstore = tcx.cstore;
    let cdata = cstore::get_crate_data(cstore, def.crate);
    decoder::get_trait_methods(cstore.intr, cdata, def.node, tcx)
}

fn get_method_names_if_trait(cstore: cstore::cstore, def: ast::def_id)
    -> Option<@DVec<(ast::ident, ast::self_ty_)>> {

    let cdata = cstore::get_crate_data(cstore, def.crate);
    return decoder::get_method_names_if_trait(cstore.intr, cdata, def.node);
}

fn get_item_attrs(cstore: cstore::cstore,
                  def_id: ast::def_id,
                  f: fn(~[@ast::meta_item])) {

    let cdata = cstore::get_crate_data(cstore, def_id.crate);
    decoder::get_item_attrs(cdata, def_id.node, f)
}

fn get_class_fields(tcx: ty::ctxt, def: ast::def_id) -> ~[ty::field_ty] {
    let cstore = tcx.cstore;
    let cdata = cstore::get_crate_data(cstore, def.crate);
    decoder::get_class_fields(cstore.intr, cdata, def.node)
}

fn get_type(tcx: ty::ctxt, def: ast::def_id) -> ty::ty_param_bounds_and_ty {
    let cstore = tcx.cstore;
    let cdata = cstore::get_crate_data(cstore, def.crate);
    decoder::get_type(cdata, def.node, tcx)
}

fn get_region_param(cstore: metadata::cstore::cstore,
                    def: ast::def_id) -> Option<ty::region_variance> {
    let cdata = cstore::get_crate_data(cstore, def.crate);
    return decoder::get_region_param(cdata, def.node);
}

fn get_field_type(tcx: ty::ctxt, class_id: ast::def_id,
                  def: ast::def_id) -> ty::ty_param_bounds_and_ty {
    let cstore = tcx.cstore;
    let cdata = cstore::get_crate_data(cstore, class_id.crate);
    let all_items = ebml::get_doc(ebml::Doc(cdata.data), tag_items);
    debug!("Looking up %?", class_id);
    let class_doc = expect(tcx.diag,
                           decoder::maybe_find_item(class_id.node, all_items),
                           || fmt!("get_field_type: class ID %? not found",
                                   class_id) );
    debug!("looking up %? : %?", def, class_doc);
    let the_field = expect(tcx.diag,
        decoder::maybe_find_item(def.node, class_doc),
        || fmt!("get_field_type: in class %?, field ID %? not found",
                 class_id, def) );
    debug!("got field data %?", the_field);
    let ty = decoder::item_type(def, the_field, tcx, cdata);
    return {bounds: @~[],
            region_param: None,
            ty: ty};
}

// Given a def_id for an impl or class, return the traits it implements,
// or the empty vector if it's not for an impl or for a class that implements
// traits
fn get_impl_traits(tcx: ty::ctxt, def: ast::def_id) -> ~[ty::t] {
    let cstore = tcx.cstore;
    let cdata = cstore::get_crate_data(cstore, def.crate);
    decoder::get_impl_traits(cdata, def.node, tcx)
}

fn get_impl_method(cstore: cstore::cstore,
                   def: ast::def_id, mname: ast::ident)
    -> ast::def_id {
    let cdata = cstore::get_crate_data(cstore, def.crate);
    decoder::get_impl_method(cstore.intr, cdata, def.node, mname)
}

/* Because classes use the trait format rather than the impl format
   for their methods (so that get_trait_methods can be reused to get
   class methods), classes require a slightly different version of
   get_impl_method. Sigh. */
fn get_class_method(cstore: cstore::cstore,
                    def: ast::def_id, mname: ast::ident)
    -> ast::def_id {
    let cdata = cstore::get_crate_data(cstore, def.crate);
    decoder::get_class_method(cstore.intr, cdata, def.node, mname)
}

/* If def names a class with a dtor, return it. Otherwise, return none. */
fn class_dtor(cstore: cstore::cstore, def: ast::def_id)
    -> Option<ast::def_id> {
    let cdata = cstore::get_crate_data(cstore, def.crate);
    decoder::class_dtor(cdata, def.node)
}
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
