import syntax::{ast, ast_util, codemap, ast_map};
import syntax::ast::*;
import ast::{ident, fn_ident, def, def_id, node_id};
import ast::{required, provided};
import syntax::ast_util::{local_def, def_id_of_def, new_def_hash,
                          class_item_ident, path_to_ident};
import pat_util::*;

import syntax::attr;
import metadata::{csearch, cstore};
import driver::session::session;
import util::common::is_main_name;
import std::map::{int_hash, str_hash, box_str_hash, hashmap};
import vec::each;
import syntax::codemap::span;
import syntax::visit;
import visit::vt;
import std::{list};
import std::list::{list, nil, cons};
import option::{is_none, is_some};
import syntax::print::pprust::*;
import dvec::{dvec, extensions};

export resolve_crate;
export def_map, ext_map, exp_map, impl_map;
export _impl, iscopes, method_info;

// Resolving happens in two passes. The first pass collects defids of all
// (internal) imports and modules, so that they can be looked up when needed,
// and then uses this information to resolve the imports. The second pass
// locates all names (in expressions, types, and alt patterns) and resolves
// them, storing the resulting def in the AST nodes.

/* foreign modules can't contain enums, and we don't store their ASTs because
   we only need to look at them to determine exports, which they can't
   control.*/

type def_map = hashmap<node_id, def>;
type ext_map = hashmap<def_id, ~[ident]>;
type impl_map = hashmap<node_id, iscopes>;
type impl_cache = hashmap<def_id, option<@~[@_impl]>>;


// Impl resolution

type method_info = {did: def_id, n_tps: uint, ident: ast::ident};
/* An _impl represents an implementation that's currently in scope.
   Its fields:
   * did: the def id of the class or impl item
   * ident: the name of the impl, unless it has no name (as in
   "impl of X") in which case the ident
   is the ident of the trait that's being implemented
   * methods: the item's methods
*/
type _impl = {did: def_id, ident: ast::ident, methods: ~[@method_info]};
type iscopes = @list<@~[@_impl]>;

type exp = {reexp: bool, id: def_id};
type exp_map = hashmap<node_id, ~[exp]>;

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
