
import syntax::{ast, ast_util, codemap};
import syntax::ast::*;
import ast::{ident, fn_ident, def, def_id, node_id};
import syntax::ast_util::{local_def, def_id_of_def};

import metadata::{csearch, cstore};
import driver::session::session;
import util::common::*;
import std::map::{new_int_hash, new_str_hash};
import syntax::codemap::span;
import syntax::visit;
import visit::vt;
import core::{vec, option, str};
import std::list;
import std::map::hashmap;
import std::list::{list, nil, cons};
import option::{some, none, is_none, is_some};
import syntax::print::pprust::*;

export resolve_crate;
export def_map, ext_map, exp_map, impl_map, iscopes;

// Resolving happens in two passes. The first pass collects defids of all
// (internal) imports and modules, so that they can be looked up when needed,
// and then uses this information to resolve the imports. The second pass
// locates all names (in expressions, types, and alt patterns) and resolves
// them, storing the resulting def in the AST nodes.

const crate_mod: int = -1;

tag scope {
    scope_crate;
    scope_item(@ast::item);
    scope_fn(ast::fn_decl, ast::proto, [ast::ty_param]);
    scope_native_item(@ast::native_item);
    scope_loop(@ast::local); // there's only 1 decl per loop.
    scope_block(ast::blk, @mutable uint, @mutable uint);
    scope_arm(ast::arm);
}

type scopes = list<scope>;

tag import_state {
    todo(ast::node_id, ast::ident, @[ast::ident], codemap::span, scopes);
    is_glob(@[ast::ident], scopes, codemap::span);
    resolving(span);
    resolved(option::t<def>, /* value */
             option::t<def>, /* type */
             option::t<def>, /* module */
             @[@ast::item],
             /* used for reporting unused import warning */
             ast::ident, codemap::span);
}

tag glob_import_state {
    glob_resolving(span);
    glob_resolved(option::t<def>,  /* value */
                  option::t<def>,  /* type */
                  option::t<def>); /* module */
}

type ext_hash = hashmap<{did: def_id, ident: str, ns: namespace}, def>;

fn new_ext_hash() -> ext_hash {
    type key = {did: def_id, ident: str, ns: namespace};
    fn hash(v: key) -> uint {
        ret str::hash(v.ident) + util::common::hash_def(v.did) +
                alt v.ns {
                  ns_value. { 1u }
                  ns_type. { 2u }
                  ns_module. { 3u }
                };
    }
    fn eq(v1: key, v2: key) -> bool {
        ret util::common::def_eq(v1.did, v2.did) &&
                str::eq(v1.ident, v2.ident) && v1.ns == v2.ns;
    }
    ret std::map::mk_hashmap::<key, def>(hash, eq);
}

tag mod_index_entry {
    mie_view_item(@ast::view_item);
    mie_import_ident(node_id, codemap::span);
    mie_item(@ast::item);
    mie_native_item(@ast::native_item);
    mie_tag_variant(/* tag item */@ast::item, /* variant index */uint);
}

type mod_index = hashmap<ident, list<mod_index_entry>>;

// A tuple of an imported def and the import stmt that brung it
type glob_imp_def = {def: def, item: @ast::view_item};

type indexed_mod = {
    m: option::t<ast::_mod>,
    index: mod_index,
    mutable glob_imports: [glob_imp_def],
    glob_imported_names: hashmap<str, glob_import_state>,
    path: str
};

/* native modules can't contain tags, and we don't store their ASTs because we
   only need to look at them to determine exports, which they can't control.*/

type def_map = hashmap<node_id, def>;
type ext_map = hashmap<def_id, [ident]>;
type exp_map = hashmap<str, def>;
type impl_map = hashmap<node_id, iscopes>;

type env =
    {cstore: cstore::cstore,
     def_map: def_map,
     ast_map: ast_map::map,
     imports: hashmap<ast::node_id, import_state>,
     exp_map: exp_map,
     mod_map: hashmap<ast::node_id, @indexed_mod>,
     block_map: hashmap<ast::node_id, [glob_imp_def]>,
     ext_map: ext_map,
     impl_map: impl_map,
     ext_cache: ext_hash,
     used_imports: {mutable track: bool,
                    mutable data: [ast::node_id]},
     mutable reported: [{ident: str, sc: scope}],
     mutable ignored_imports: [node_id],
     sess: session};


// Used to distinguish between lookups from outside and from inside modules,
// since export restrictions should only be applied for the former.
tag dir { inside; outside; }

tag namespace { ns_value; ns_type; ns_module; }

fn resolve_crate(sess: session, amap: ast_map::map, crate: @ast::crate) ->
   {def_map: def_map, ext_map: ext_map,
    exp_map: exp_map, impl_map: impl_map} {
    let e =
        @{cstore: sess.get_cstore(),
          def_map: new_int_hash(),
          ast_map: amap,
          imports: new_int_hash(),
          exp_map: new_str_hash(),
          mod_map: new_int_hash(),
          block_map: new_int_hash(),
          ext_map: new_def_hash(),
          impl_map: new_int_hash(),
          ext_cache: new_ext_hash(),
          used_imports: {mutable track: false, mutable data:  []},
          mutable reported: [],
          mutable ignored_imports: [],
          sess: sess};
    map_crate(e, crate);
    resolve_imports(*e);
    check_for_collisions(e, *crate);
    check_exports(e);
    resolve_names(e, crate);
    resolve_impls(e, crate);
    if sess.get_opts().warn_unused_imports {
        check_unused_imports(e);
    }
    ret {def_map: e.def_map, ext_map: e.ext_map,
         exp_map: e.exp_map, impl_map: e.impl_map};
}

// Locate all modules and imports and index them, so that the next passes can
// resolve through them.
fn map_crate(e: @env, c: @ast::crate) {
    // First, find all the modules, and index the names that they contain
    let v_map_mod =
        @{visit_view_item: bind index_vi(e, _, _, _),
          visit_item: bind index_i(e, _, _, _),
          visit_block: visit_block_with_scope
          with *visit::default_visitor::<scopes>()};
    visit::visit_crate(*c, cons(scope_crate, @nil), visit::mk_vt(v_map_mod));

    // Register the top-level mod
    e.mod_map.insert(crate_mod,
                     @{m: some(c.node.module),
                       index: index_mod(c.node.module),
                       mutable glob_imports: [],
                       glob_imported_names: new_str_hash(),
                       path: ""});
    fn index_vi(e: @env, i: @ast::view_item, sc: scopes, _v: vt<scopes>) {
        alt i.node {
          ast::view_item_import(name, ids, id) {
            e.imports.insert(id, todo(id, name, ids, i.span, sc));
          }
          ast::view_item_import_from(mod_path, idents, id) {
            for ident in idents {
                e.imports.insert(ident.node.id,
                                 todo(ident.node.id, ident.node.name,
                                      @(*mod_path + [ident.node.name]),
                                      ident.span, sc));
            }
          }
          ast::view_item_import_glob(pth, id) {
            e.imports.insert(id, is_glob(pth, sc, i.span));
          }
          _ { }
        }
    }
    fn path_from_scope(sc: scopes, n: str) -> str {
        let path = n + "::";
        list::iter(sc) {|s|
            alt s {
              scope_item(i) { path = i.ident + "::" + path; }
              _ {}
            }
        }
        path
    }
    fn index_i(e: @env, i: @ast::item, sc: scopes, v: vt<scopes>) {
        visit_item_with_scope(i, sc, v);
        alt i.node {
          ast::item_mod(md) {
            e.mod_map.insert(i.id,
                             @{m: some(md),
                               index: index_mod(md),
                               mutable glob_imports: [],
                               glob_imported_names: new_str_hash(),
                               path: path_from_scope(sc, i.ident)});
          }
          ast::item_native_mod(nmd) {
            e.mod_map.insert(i.id,
                             @{m: none::<ast::_mod>,
                               index: index_nmod(nmd),
                               mutable glob_imports: [],
                               glob_imported_names: new_str_hash(),
                               path: path_from_scope(sc, i.ident)});
          }
          _ { }
        }
    }

    // Next, assemble the links for globbed imports.
    let v_link_glob =
        @{visit_view_item: bind link_glob(e, _, _, _),
          visit_block: visit_block_with_scope,
          visit_item: visit_item_with_scope
          with *visit::default_visitor::<scopes>()};
    visit::visit_crate(*c, cons(scope_crate, @nil),
                       visit::mk_vt(v_link_glob));
    fn link_glob(e: @env, vi: @ast::view_item, sc: scopes, _v: vt<scopes>) {
        alt vi.node {
          //if it really is a glob import, that is
          ast::view_item_import_glob(path, _) {
            let imp = follow_import(*e, sc, *path, vi.span);
            if option::is_some(imp) {
                let glob = {def: option::get(imp), item: vi};
                alt list::head(sc) {
                  scope_item(i) {
                    e.mod_map.get(i.id).glob_imports += [glob];
                  }
                  scope_block(b, _, _) {
                    let globs = alt e.block_map.find(b.node.id) {
                      some(globs) { globs + [glob] } none. { [glob] }
                    };
                    e.block_map.insert(b.node.id, globs);
                  }
                  scope_crate. {
                    e.mod_map.get(crate_mod).glob_imports += [glob];
                  }
                }
            }
          }
          _ { }
        }
    }
}

fn resolve_imports(e: env) {
    e.used_imports.track = true;
    e.imports.values {|v|
        alt v {
          todo(node_id, name, path, span, scopes) {
            resolve_import(e, local_def(node_id), name, *path, span, scopes);
          }
          resolved(_, _, _, _, _, _) | is_glob(_, _, _) { }
        }
    };
    e.used_imports.track = false;
    e.sess.abort_if_errors();
}

fn check_unused_imports(e: @env) {
    e.imports.items {|k, v|
        alt v {
            resolved(_, _, _, _, name, sp) {
              if !vec::member(k, e.used_imports.data) {
                e.sess.span_warn(sp, "unused import " + name);
              }
            }
            _ { }
        }
    };
}

fn resolve_names(e: @env, c: @ast::crate) {
    e.used_imports.track = true;
    let v =
        @{visit_native_item: visit_native_item_with_scope,
          visit_item: visit_item_with_scope,
          visit_block: visit_block_with_scope,
          visit_decl: visit_decl_with_scope,
          visit_arm: visit_arm_with_scope,
          visit_pat: bind walk_pat(e, _, _, _),
          visit_expr: bind walk_expr(e, _, _, _),
          visit_ty: bind walk_ty(e, _, _, _),
          visit_constr: bind walk_constr(e, _, _, _, _, _),
          visit_fn: bind visit_fn_with_scope(e, _, _, _, _, _, _, _)
             with *visit::default_visitor()};
    visit::visit_crate(*c, cons(scope_crate, @nil), visit::mk_vt(v));
    e.used_imports.track = false;
    e.sess.abort_if_errors();

    fn walk_expr(e: @env, exp: @ast::expr, sc: scopes, v: vt<scopes>) {
        visit_expr_with_scope(exp, sc, v);
        alt exp.node {
          ast::expr_path(p) {
            maybe_insert(e, exp.id,
                         lookup_path_strict(*e, sc, exp.span, p.node,
                                            ns_value));
          }
          _ { }
        }
    }
    fn walk_ty(e: @env, t: @ast::ty, sc: scopes, v: vt<scopes>) {
        visit::visit_ty(t, sc, v);
        alt t.node {
          ast::ty_path(p, id) {
            maybe_insert(e, id,
                         lookup_path_strict(*e, sc, t.span, p.node, ns_type));
          }
          _ { }
        }
    }
    fn walk_constr(e: @env, p: @ast::path, sp: span, id: node_id, sc: scopes,
                   _v: vt<scopes>) {
        maybe_insert(e, id, lookup_path_strict(*e, sc, sp, p.node, ns_value));
    }
    fn walk_pat(e: @env, pat: @ast::pat, sc: scopes, v: vt<scopes>) {
        visit::visit_pat(pat, sc, v);
        alt pat.node {
          ast::pat_tag(p, _) {
            let fnd = lookup_path_strict(*e, sc, p.span, p.node, ns_value);
            alt option::get(fnd) {
              ast::def_variant(did, vid) {
                e.def_map.insert(pat.id, option::get(fnd));
              }
              _ {
                e.sess.span_err(p.span,
                                "not a tag variant: " +
                                    ast_util::path_name(p));
              }
            }
          }
          _ { }
        }
    }

    fn maybe_insert(e: @env, id: node_id, def: option::t<def>) {
        if option::is_some(def) { e.def_map.insert(id, option::get(def)); }
    }
}


// Visit helper functions
fn visit_item_with_scope(i: @ast::item, sc: scopes, v: vt<scopes>) {
    visit::visit_item(i, cons(scope_item(i), @sc), v);
}

fn visit_native_item_with_scope(ni: @ast::native_item, sc: scopes,
                                v: vt<scopes>) {
    visit::visit_native_item(ni, cons(scope_native_item(ni), @sc), v);
}

fn visit_fn_with_scope(e: @env, f: ast::_fn, tp: [ast::ty_param], sp: span,
                       name: fn_ident, id: node_id, sc: scopes,
                       v: vt<scopes>) {
    // is this a main fn declaration?
    alt name {
      some(nm) {
        if is_main_name([nm]) && !e.sess.building_library() {
            // This is a main function -- set it in the session
            // as the main ID
            e.sess.set_main_id(id);
        }
      }
      _ { }
    }

    // here's where we need to set up the mapping
    // for f's constrs in the table.
    for c: @ast::constr in f.decl.constraints { resolve_constr(e, c, sc, v); }
    visit::visit_fn(f, tp, sp, name, id,
                    cons(scope_fn(f.decl, f.proto, tp), @sc), v);
}

fn visit_block_with_scope(b: ast::blk, sc: scopes, v: vt<scopes>) {
    let pos = @mutable 0u, loc = @mutable 0u;
    let block_sc = cons(scope_block(b, pos, loc), @sc);
    for vi in b.node.view_items { v.visit_view_item(vi, block_sc, v); }
    for stmt in b.node.stmts {
        v.visit_stmt(stmt, block_sc, v);;
        *pos += 1u;;
        *loc = 0u;
    }
    visit::visit_expr_opt(b.node.expr, block_sc, v);
}

fn visit_decl_with_scope(d: @decl, sc: scopes, v: vt<scopes>) {
    let loc_pos = alt list::head(sc) {
      scope_block(_, _, pos) { pos }
      _ { @mutable 0u }
    };
    alt d.node {
      decl_local(locs) {
        for (_, loc) in locs { v.visit_local(loc, sc, v);; *loc_pos += 1u; }
      }
      decl_item(it) { v.visit_item(it, sc, v); }
    }
}

fn visit_arm_with_scope(a: ast::arm, sc: scopes, v: vt<scopes>) {
    for p: @pat in a.pats { v.visit_pat(p, sc, v); }
    let sc_inner = cons(scope_arm(a), @sc);
    visit::visit_expr_opt(a.guard, sc_inner, v);
    v.visit_block(a.body, sc_inner, v);
}

fn visit_expr_with_scope(x: @ast::expr, sc: scopes, v: vt<scopes>) {
    alt x.node {
      ast::expr_for(decl, coll, blk) {
        let new_sc = cons(scope_loop(decl), @sc);
        v.visit_expr(coll, sc, v);
        v.visit_local(decl, new_sc, v);
        v.visit_block(blk, new_sc, v);
      }
      _ { visit::visit_expr(x, sc, v); }
    }
}

fn follow_import(e: env, sc: scopes, path: [ident], sp: span) ->
   option::t<def> {
    let path_len = vec::len(path);
    let dcur = lookup_in_scope_strict(e, sc, sp, path[0], ns_module);
    let i = 1u;
    while true && option::is_some(dcur) {
        if i == path_len { break; }
        dcur =
            lookup_in_mod_strict(e, option::get(dcur), sp, path[i],
                                 ns_module, outside);
        i += 1u;
    }
    if i == path_len {
        alt option::get(dcur) {
          ast::def_mod(_) | ast::def_native_mod(_) { ret dcur; }
          _ {
            e.sess.span_err(sp, str::connect(path, "::") +
                            " does not name a module.");
            ret none;
          }
        }
    } else { ret none; }
}

fn resolve_constr(e: @env, c: @ast::constr, sc: scopes, _v: vt<scopes>) {
    let new_def =
        lookup_path_strict(*e, sc, c.span, c.node.path.node, ns_value);
    if option::is_some(new_def) {
        alt option::get(new_def) {
          ast::def_fn(pred_id, ast::pure_fn.) {
            e.def_map.insert(c.node.id, ast::def_fn(pred_id, ast::pure_fn));
          }
          _ {
            e.sess.span_err(c.span,
                            "Non-predicate in constraint: " +
                                path_to_str(c.node.path));
          }
        }
    }
}

// Import resolution
fn resolve_import(e: env, defid: ast::def_id, name: ast::ident,
                  ids: [ast::ident], sp: codemap::span, sc: scopes) {
    fn register(e: env, id: node_id, cx: ctxt, sp: codemap::span,
                name: ast::ident, lookup: block(namespace) -> option::t<def>,
                impls: [@ast::item]) {
        let val = lookup(ns_value), typ = lookup(ns_type),
            md = lookup(ns_module);
        if is_none(val) && is_none(typ) && is_none(md) {
            unresolved_err(e, cx, sp, name, "import");
        } else {
            e.imports.insert(id, resolved(val, typ, md, @impls, name, sp));
        }
    }
    // Temporarily disable this import and the imports coming after during
    // resolution of this import.
    fn find_imports_after(e: env, id: node_id, sc: scopes) -> [node_id] {
        fn lst(my_id: node_id, vis: [@view_item]) -> [node_id] {
            let imports = [], found = false;
            for vi in vis {
                alt vi.node {
                  view_item_import(_, _, id) | view_item_import_glob(_, id) {
                    if id == my_id { found = true; }
                    if found { imports += [id]; }
                  }
                  view_item_import_from(_, ids, _) {
                    for id in ids {
                        if id.node.id == my_id { found = true; }
                        if found { imports += [id.node.id]; }
                    }
                  }
                  _ {}
                }
            }
            imports
        }
        alt sc {
          cons(scope_item(@{node: item_mod(m), _}), _) {
            lst(id, m.view_items)
          }
          cons(scope_item(@{node: item_native_mod(m), _}), _) {
            lst(id, m.view_items)
          }
          cons(scope_block(b, _, _), _) {
            lst(id, b.node.view_items)
          }
          cons(scope_crate., _) {
            lst(id, option::get(e.mod_map.get(crate_mod).m).view_items)
          }
        }
    }
    // This function has cleanup code at the end. Do not return without going
    // through that.
    e.imports.insert(defid.node, resolving(sp));
    let ignored = find_imports_after(e, defid.node, sc);
    e.ignored_imports <-> ignored;
    let n_idents = vec::len(ids);
    let end_id = ids[n_idents - 1u];
    if n_idents == 1u {
        register(e, defid.node, in_scope(sc), sp, name,
                 {|ns| lookup_in_scope(e, sc, sp, end_id, ns) }, []);
    } else {
        alt lookup_in_scope(e, sc, sp, ids[0], ns_module) {
          none. {
            unresolved_err(e, in_scope(sc), sp, ids[0], ns_name(ns_module));
          }
          some(dcur_) {
            let dcur = dcur_, i = 1u;
            while true {
                if i == n_idents - 1u {
                    let impls = [];
                    find_impls_in_mod(e, dcur, impls, some(end_id));
                    register(e, defid.node, in_mod(dcur), sp, name, {|ns|
                        lookup_in_mod(e, dcur, sp, end_id, ns, outside)
                    }, impls);
                    break;
                } else {
                    dcur = alt lookup_in_mod(e, dcur, sp, ids[i], ns_module,
                                             outside) {
                      some(dcur) { dcur }
                      none. {
                        unresolved_err(e, in_mod(dcur), sp, ids[i],
                                       ns_name(ns_module));
                        break;
                      }
                    };
                    i += 1u;
                }
            }
          }
        }
    }
    e.ignored_imports <-> ignored;
    // If we couldn't resolve the import, don't leave it in a partially
    // resolved state, to avoid having it reported later as a cyclic
    // import
    alt e.imports.find(defid.node) {
      some(resolving(sp)) {
        e.imports.insert(defid.node, resolved(none, none, none, @[], "", sp));
      }
      _ { }
    }
}


// Utilities
fn ns_name(ns: namespace) -> str {
    alt ns {
      ns_type. { ret "typename"; }
      ns_value. { ret "name"; }
      ns_module. { ret "modulename"; }
    }
}

tag ctxt { in_mod(def); in_scope(scopes); }

fn unresolved_err(e: env, cx: ctxt, sp: span, name: ident, kind: str) {
    fn find_fn_or_mod_scope(sc: scopes) -> option::t<scope> {
        let sc = sc;
        while true {
            alt sc {
              cons(cur, rest) {
                alt cur {
                  scope_crate. | scope_fn(_, _, _) |
                  scope_item(@{node: ast::item_mod(_), _}) {
                    ret some(cur);
                  }
                  _ { sc = *rest; }
                }
              }
              _ { ret none; }
            }
        }
        fail;
    }
    let path = name;
    alt cx {
      in_scope(sc) {
        alt find_fn_or_mod_scope(sc) {
          some(err_scope) {
            for rs: {ident: str, sc: scope} in e.reported {
                if str::eq(rs.ident, name) && err_scope == rs.sc { ret; }
            }
            e.reported += [{ident: name, sc: err_scope}];
          }
          _ {}
        }
      }
      in_mod(def) {
        let did = def_id_of_def(def);
        if did.crate == ast::local_crate {
            path = e.mod_map.get(did.node).path + path;
        } else if did.node != -1 {
            let paths = e.ext_map.get(did);
            if vec::len(paths) > 0u {
                path = str::connect(paths, "::") + "::" + path;
            }
        }
      }
    }
    e.sess.span_err(sp, mk_unresolved_msg(path, kind));
}

fn unresolved_fatal(e: env, sp: span, id: ident, kind: str) -> ! {
    e.sess.span_fatal(sp, mk_unresolved_msg(id, kind));
}

fn mk_unresolved_msg(id: ident, kind: str) -> str {
    ret #fmt["unresolved %s: %s", kind, id];
}

// Lookup helpers
fn lookup_path_strict(e: env, sc: scopes, sp: span, pth: ast::path_,
                      ns: namespace) -> option::t<def> {
    let n_idents = vec::len(pth.idents);
    let headns = if n_idents == 1u { ns } else { ns_module };

    let first_scope;
    if pth.global {
        first_scope = list::cons(scope_crate, @list::nil);
    } else { first_scope = sc; }

    let dcur =
        lookup_in_scope_strict(e, first_scope, sp, pth.idents[0], headns);

    let i = 1u;
    while i < n_idents && option::is_some(dcur) {
        let curns = if n_idents == i + 1u { ns } else { ns_module };
        dcur =
            lookup_in_mod_strict(e, option::get(dcur), sp, pth.idents[i],
                                 curns, outside);
        i += 1u;
    }
    ret dcur;
}

fn lookup_in_scope_strict(e: env, sc: scopes, sp: span, name: ident,
                          ns: namespace) -> option::t<def> {
    alt lookup_in_scope(e, sc, sp, name, ns) {
      none. {
        unresolved_err(e, in_scope(sc), sp, name, ns_name(ns));
        ret none;
      }
      some(d) { ret some(d); }
    }
}

fn scope_is_fn(sc: scope) -> bool {
    ret alt sc {
          scope_fn(_, ast::proto_bare., _) |
          scope_native_item(_) {
            true
          }
          _ { false }
        };
}

// Returns:
//   none - does not close
//   some(true) - closes and permits mutation
//   some(false) - closes but no mutation
fn scope_closes(sc: scope) -> option::t<bool> {
    alt sc {
      scope_fn(_, ast::proto_block., _) { some(true) }
      scope_fn(_, ast::proto_send., _) { some(false) }
      scope_fn(_, ast::proto_shared(_), _) { some(false) }
      _ { none }
    }
}

fn def_is_local(d: def) -> bool {
    ret alt d {
          ast::def_arg(_, _) | ast::def_local(_, _) | ast::def_binding(_) |
          ast::def_upvar(_, _, _) {
            true
          }
          _ { false }
        };
}

fn def_is_obj_field(d: def) -> bool {
    ret alt d { ast::def_obj_field(_, _) { true } _ { false } };
}

fn def_is_ty_arg(d: def) -> bool {
    ret alt d { ast::def_ty_param(_, _) { true } _ { false } };
}

fn lookup_in_scope(e: env, sc: scopes, sp: span, name: ident, ns: namespace)
   -> option::t<def> {
    fn in_scope(e: env, sp: span, name: ident, s: scope, ns: namespace) ->
       option::t<def> {
        alt s {
          scope_crate. {
            ret lookup_in_local_mod(e, crate_mod, sp, name, ns, inside);
          }
          scope_item(it) {
            alt it.node {
              ast::item_obj(ob, ty_params, _) {
                ret lookup_in_obj(name, ob, ty_params, ns);
              }
              ast::item_impl(_, _, _) {
                if (name == "self" && ns == ns_value) {
                    ret some(ast::def_self(local_def(it.id)));
                }
              }
              ast::item_tag(_, ty_params) {
                if ns == ns_type { ret lookup_in_ty_params(name, ty_params); }
              }
              ast::item_mod(_) {
                ret lookup_in_local_mod(e, it.id, sp, name, ns, inside);
              }
              ast::item_native_mod(m) {
                ret lookup_in_local_native_mod(e, it.id, sp, name, ns);
              }
              ast::item_ty(_, ty_params) {
                if ns == ns_type { ret lookup_in_ty_params(name, ty_params); }
              }
              _ { }
            }
          }
          scope_native_item(it) {
            alt it.node {
              ast::native_item_fn(decl, ty_params) {
                ret lookup_in_fn(name, decl, ty_params, ns);
              }
            }
          }
          scope_fn(decl, _, ty_params) {
            ret lookup_in_fn(name, decl, ty_params, ns);
          }
          scope_loop(local) {
            if ns == ns_value {
                alt lookup_in_pat(name, local.node.pat) {
                  some(did) { ret some(ast::def_binding(did)); }
                  _ { }
                }
            }
          }
          scope_block(b, pos, loc) {
            ret lookup_in_block(e, name, sp, b.node, *pos, *loc, ns);
          }
          scope_arm(a) {
            if ns == ns_value {
                alt lookup_in_pat(name, a.pats[0]) {
                  some(did) { ret some(ast::def_binding(did)); }
                  _ { ret none; }
                }
            }
          }
        }
        ret none::<def>;
    }
    let left_fn = false;
    let closing = [];
    // Used to determine whether obj fields are in scope
    let left_fn_level2 = false;
    let sc = sc;
    while true {
        alt copy sc {
          nil. { ret none::<def>; }
          cons(hd, tl) {
            let fnd = in_scope(e, sp, name, hd, ns);
            if !is_none(fnd) {
                let df = option::get(fnd);
                let local = def_is_local(df);
                if left_fn && local || left_fn_level2 && def_is_obj_field(df)
                       || scope_is_fn(hd) && left_fn && def_is_ty_arg(df) {
                    let msg =
                        alt ns {
                          ns_type. {
                            "Attempt to use a type argument out of scope"
                          }
                          _ { "attempted dynamic environment-capture" }
                        };
                    e.sess.span_fatal(sp, msg);
                } else if local {
                    let i = vec::len(closing);
                    while i > 0u {
                        i -= 1u;
                        df =
                            ast::def_upvar(def_id_of_def(df), @df,
                                           closing[i]);
                        fnd = some(df);
                    }
                }
                ret fnd;
            }
            if left_fn {
                left_fn_level2 = true;
            } else if ns == ns_value || ns == ns_type {
                left_fn = scope_is_fn(hd);
                alt scope_closes(hd) { some(mut) { closing += [mut]; } _ { } }
            }
            sc = *tl;
          }
        }
    }
    e.sess.bug("reached unreachable code in lookup_in_scope"); // sigh
}

fn lookup_in_ty_params(name: ident, ty_params: [ast::ty_param]) ->
   option::t<def> {
    let i = 0u;
    for tp: ast::ty_param in ty_params {
        if str::eq(tp.ident, name) {
            ret some(ast::def_ty_param(i, ast_util::ty_param_kind(tp)));
        }
        i += 1u;
    }
    ret none::<def>;
}

fn lookup_in_pat(name: ident, pat: @ast::pat) -> option::t<def_id> {
    let found = none;
    ast_util::pat_bindings(pat) {|bound|
        let p_name = alt bound.node { ast::pat_bind(n, _) { n } };
        if str::eq(p_name, name) { found = some(local_def(bound.id)); }
    };
    ret found;
}

fn lookup_in_fn(name: ident, decl: ast::fn_decl, ty_params: [ast::ty_param],
                ns: namespace) -> option::t<def> {
    alt ns {
      ns_value. {
        for a: ast::arg in decl.inputs {
            if str::eq(a.ident, name) {
                ret some(ast::def_arg(local_def(a.id), a.mode));
            }
        }
        ret none::<def>;
      }
      ns_type. { ret lookup_in_ty_params(name, ty_params); }
      _ { ret none::<def>; }
    }
}

fn lookup_in_obj(name: ident, ob: ast::_obj, ty_params: [ast::ty_param],
                 ns: namespace) -> option::t<def> {
    alt ns {
      ns_value. {
        for f: ast::obj_field in ob.fields {
            if str::eq(f.ident, name) {
                ret some(ast::def_obj_field(local_def(f.id), f.mut));
            }
        }
        ret none::<def>;
      }
      ns_type. { ret lookup_in_ty_params(name, ty_params); }
      _ { ret none::<def>; }
    }
}

fn lookup_in_block(e: env, name: ident, sp: span, b: ast::blk_, pos: uint,
                   loc_pos: uint, ns: namespace) -> option::t<def> {
    let i = vec::len(b.stmts);
    while i > 0u {
        i -= 1u;
        let st = b.stmts[i];
        alt st.node {
          ast::stmt_decl(d, _) {
            alt d.node {
              ast::decl_local(locs) {
                if i <= pos {
                    let j = vec::len(locs);
                    while j > 0u {
                        j -= 1u;
                        let (style, loc) = locs[j];
                        if ns == ns_value && (i < pos || j < loc_pos) {
                            alt lookup_in_pat(name, loc.node.pat) {
                              some(did) {
                                ret some(ast::def_local(did, style));
                              }
                              _ { }
                            }
                        }
                    }
                }
              }
              ast::decl_item(it) {
                alt it.node {
                  ast::item_tag(variants, _) {
                    if ns == ns_type {
                        if str::eq(it.ident, name) {
                            ret some(ast::def_ty(local_def(it.id)));
                        }
                    } else if ns == ns_value {
                        for v: ast::variant in variants {
                            if str::eq(v.node.name, name) {
                                let i = v.node.id;
                                ret some(ast::def_variant(local_def(it.id),
                                                          local_def(i)));
                            }
                        }
                    }
                  }
                  _ {
                    if str::eq(it.ident, name) {
                        let found = found_def_item(it, ns);
                        if !is_none(found) { ret found; }
                    }
                  }
                }
              }
            }
          }
          _ { }
        }
    }
    for vi in b.view_items {
        alt vi.node {
          ast::view_item_import(ident, _, id) {
            if name == ident { ret lookup_import(e, local_def(id), ns); }
          }
          ast::view_item_import_from(mod_path, idents, id) {
            for ident in idents {
                if name == ident.node.name {
                    ret lookup_import(e, local_def(ident.node.id), ns);
                }
            }
          }
          ast::view_item_import_glob(_, _) {
            alt e.block_map.find(b.id) {
              some(globs) {
                let found = lookup_in_globs(e, globs, sp, name, ns, inside);
                if found != none { ret found; }
              }
              _ {}
            }
          }
        }
    }
    ret none;
}

fn found_def_item(i: @ast::item, ns: namespace) -> option::t<def> {
    alt i.node {
      ast::item_const(_, _) {
        if ns == ns_value { ret some(ast::def_const(local_def(i.id))); }
      }
      ast::item_fn(f, _) {
        if ns == ns_value {
            ret some(ast::def_fn(local_def(i.id), f.decl.purity));
        }
      }
      ast::item_mod(_) {
        if ns == ns_module { ret some(ast::def_mod(local_def(i.id))); }
      }
      ast::item_native_mod(_) {
        if ns == ns_module { ret some(ast::def_native_mod(local_def(i.id))); }
      }
      ast::item_ty(_, _) {
        if ns == ns_type { ret some(ast::def_ty(local_def(i.id))); }
      }
      ast::item_res(_, _, _, ctor_id) {
        alt ns {
          ns_value. {
            ret some(ast::def_fn(local_def(ctor_id), ast::impure_fn));
          }
          ns_type. { ret some(ast::def_ty(local_def(i.id))); }
          _ { }
        }
      }
      ast::item_tag(_, _) {
        if ns == ns_type { ret some(ast::def_ty(local_def(i.id))); }
      }
      ast::item_obj(_, _, ctor_id) {
        alt ns {
          ns_value. {
            ret some(ast::def_fn(local_def(ctor_id), ast::impure_fn));
          }
          ns_type. { ret some(ast::def_ty(local_def(i.id))); }
          _ { }
        }
      }
      _ { }
    }
    ret none;
}

fn lookup_in_mod_strict(e: env, m: def, sp: span, name: ident,
                        ns: namespace, dr: dir) -> option::t<def> {
    alt lookup_in_mod(e, m, sp, name, ns, dr) {
      none. {
        unresolved_err(e, in_mod(m), sp, name, ns_name(ns));
        ret none;
      }
      some(d) { ret some(d); }
    }
}

fn lookup_in_mod(e: env, m: def, sp: span, name: ident, ns: namespace,
                 dr: dir) -> option::t<def> {
    let defid = def_id_of_def(m);
    if defid.crate != ast::local_crate {
        // examining a module in an external crate
        let cached = e.ext_cache.find({did: defid, ident: name, ns: ns});
        if !is_none(cached) { ret cached; }
        let path = [name];
        if defid.node != crate_mod { path = e.ext_map.get(defid) + path; }
        let fnd = lookup_external(e, defid.crate, path, ns);
        if !is_none(fnd) {
            e.ext_cache.insert({did: defid, ident: name, ns: ns},
                               option::get(fnd));
        }
        ret fnd;
    }
    alt m {
      ast::def_mod(defid) {
        ret lookup_in_local_mod(e, defid.node, sp, name, ns, dr);
      }
      ast::def_native_mod(defid) {
        ret lookup_in_local_native_mod(e, defid.node, sp, name, ns);
      }
    }
}

fn found_view_item(e: env, vi: @ast::view_item) -> option::t<def> {
    alt vi.node {
      ast::view_item_use(_, _, id) {
        let cnum = cstore::get_use_stmt_cnum(e.cstore, id);
        ret some(ast::def_mod({crate: cnum, node: crate_mod}));
      }
    }
}

fn lookup_import(e: env, defid: def_id, ns: namespace) -> option::t<def> {
    // Imports are simply ignored when resolving themselves.
    if vec::member(defid.node, e.ignored_imports) { ret none; }
    alt e.imports.get(defid.node) {
      todo(node_id, name, path, span, scopes) {
        resolve_import(e, local_def(node_id), name, *path, span, scopes);
        ret lookup_import(e, defid, ns);
      }
      resolving(sp) {
        e.sess.span_err(sp, "cyclic import");
        ret none;
      }
      resolved(val, typ, md, _, _, _) {
        if e.used_imports.track {
            e.used_imports.data += [defid.node];
        }
        ret alt ns { ns_value. { val } ns_type. { typ }
                     ns_module. { md } };
      }
    }
}

fn lookup_in_local_native_mod(e: env, node_id: node_id, sp: span, id: ident,
                              ns: namespace) -> option::t<def> {
    ret lookup_in_local_mod(e, node_id, sp, id, ns, inside);
}

fn lookup_in_local_mod(e: env, node_id: node_id, sp: span, id: ident,
                       ns: namespace, dr: dir) -> option::t<def> {
    let info = e.mod_map.get(node_id);
    if dr == outside && !ast_util::is_exported(id, option::get(info.m)) {
        // if we're in a native mod, then dr==inside, so info.m is some _mod
        ret none::<def>; // name is not visible
    }
    alt info.index.find(id) {
      none. { }
      some(lst_) {
        let lst = lst_;
        while true {
            alt lst {
              nil. { break; }
              cons(hd, tl) {
                let found = lookup_in_mie(e, hd, ns);
                if !is_none(found) { ret found; }
                lst = *tl;
              }
            }
        }
      }
    }
    // not local or explicitly imported; try globs:
    ret lookup_glob_in_mod(e, info, sp, id, ns, outside);
}

fn lookup_in_globs(e: env, globs: [glob_imp_def], sp: span, id: ident,
                   ns: namespace, dr: dir) -> option::t<def> {
    fn lookup_in_mod_(e: env, def: glob_imp_def, sp: span, name: ident,
                      ns: namespace, dr: dir) -> option::t<glob_imp_def> {
        alt def.item.node {
          ast::view_item_import_glob(_, id) {
            if vec::member(id, e.ignored_imports) { ret none; }
          }
        }
        alt lookup_in_mod(e, def.def, sp, name, ns, dr) {
          some(d) { option::some({def: d, item: def.item}) }
          none. { none }
        }
    }
    let matches = vec::filter_map(bind lookup_in_mod_(e, _, sp, id, ns, dr),
                                  copy globs);
    if vec::len(matches) == 0u {
        ret none;
    } else if vec::len(matches) == 1u {
        ret some(matches[0].def);
    } else {
        for match: glob_imp_def in matches {
            let sp = match.item.span;
            e.sess.span_note(sp, #fmt["'%s' is imported here", id]);
        }
        e.sess.span_fatal(sp, "'" + id + "' is glob-imported from" +
                          " multiple different modules.");
    }
}

fn lookup_glob_in_mod(e: env, info: @indexed_mod, sp: span, id: ident,
                      wanted_ns: namespace, dr: dir) -> option::t<def> {
    // since we don't know what names we have in advance,
    // absence takes the place of todo()
    if !info.glob_imported_names.contains_key(id) {
        info.glob_imported_names.insert(id, glob_resolving(sp));
        let val = lookup_in_globs(e, info.glob_imports, sp, id, ns_value, dr);
        let typ = lookup_in_globs(e, info.glob_imports, sp, id, ns_type, dr);
        let md = lookup_in_globs(e, info.glob_imports, sp, id, ns_module, dr);
        info.glob_imported_names.insert(id, glob_resolved(val, typ, md));
    }
    alt info.glob_imported_names.get(id) {
      glob_resolving(sp) { ret none::<def>; }
      glob_resolved(val, typ, md) {
        ret alt wanted_ns {
          ns_value. { val }
          ns_type. { typ }
          ns_module. { md }
        };
      }
    }
}

fn lookup_in_mie(e: env, mie: mod_index_entry, ns: namespace) ->
   option::t<def> {
    alt mie {
      mie_view_item(view_item) {
        if ns == ns_module { ret found_view_item(e, view_item); }
      }
      mie_import_ident(id, _) { ret lookup_import(e, local_def(id), ns); }
      mie_item(item) { ret found_def_item(item, ns); }
      mie_tag_variant(item, variant_idx) {
        alt item.node {
          ast::item_tag(variants, _) {
            if ns == ns_value {
                let vid = variants[variant_idx].node.id;
                ret some(ast::def_variant(local_def(item.id),
                                          local_def(vid)));
            } else { ret none::<def>; }
          }
        }
      }
      mie_native_item(native_item) {
        alt native_item.node {
          ast::native_item_ty. {
            if ns == ns_type {
                ret some(ast::def_native_ty(local_def(native_item.id)));
            }
          }
          ast::native_item_fn(decl, _) {
            if ns == ns_value {
                ret some(ast::def_native_fn(
                    local_def(native_item.id),
                    decl.purity));
            }
          }
        }
      }
    }
    ret none::<def>;
}


// Module indexing
fn add_to_index(index: hashmap<ident, list<mod_index_entry>>, id: ident,
                ent: mod_index_entry) {
    alt index.find(id) {
      none. { index.insert(id, cons(ent, @nil::<mod_index_entry>)); }
      some(prev) { index.insert(id, cons(ent, @prev)); }
    }
}

fn index_mod(md: ast::_mod) -> mod_index {
    let index = new_str_hash::<list<mod_index_entry>>();
    for it: @ast::view_item in md.view_items {
        alt it.node {
          ast::view_item_use(ident, _, _) {
            add_to_index(index, ident, mie_view_item(it));
          }
          ast::view_item_import(ident, _, id) {
            add_to_index(index, ident, mie_import_ident(id, it.span));
          }
          ast::view_item_import_from(_, idents, _) {
            for ident in idents {
                add_to_index(index, ident.node.name,
                             mie_import_ident(ident.node.id, ident.span));
            }
          }
          //globbed imports have to be resolved lazily.
          ast::view_item_import_glob(_, _) | ast::view_item_export(_, _) {}
        }
    }
    for it: @ast::item in md.items {
        alt it.node {
          ast::item_const(_, _) | ast::item_fn(_, _) | ast::item_mod(_) |
          ast::item_native_mod(_) | ast::item_ty(_, _) |
          ast::item_res(_, _, _, _) | ast::item_obj(_, _, _) |
          ast::item_impl(_, _, _) {
            add_to_index(index, it.ident, mie_item(it));
          }
          ast::item_tag(variants, _) {
            add_to_index(index, it.ident, mie_item(it));
            let variant_idx: uint = 0u;
            for v: ast::variant in variants {
                add_to_index(index, v.node.name,
                             mie_tag_variant(it, variant_idx));
                variant_idx += 1u;
            }
          }
        }
    }
    ret index;
}

fn index_nmod(md: ast::native_mod) -> mod_index {
    let index = new_str_hash::<list<mod_index_entry>>();
    for it: @ast::view_item in md.view_items {
        alt it.node {
          ast::view_item_use(ident, _, _) {
            add_to_index(index, ident, mie_view_item(it));
          }
          ast::view_item_import(ident, _, id) {
            add_to_index(index, ident, mie_import_ident(id, it.span));
          }
          ast::view_item_import_from(_, idents, _) {
            for ident in idents {
                add_to_index(index, ident.node.name,
                             mie_import_ident(ident.node.id, ident.span));
            }
          }
          ast::view_item_import_glob(_, _) | ast::view_item_export(_, _) { }
        }
    }
    for it: @ast::native_item in md.items {
        add_to_index(index, it.ident, mie_native_item(it));
    }
    ret index;
}


// External lookups
fn ns_for_def(d: def) -> namespace {
    ret alt d {
          ast::def_fn(_, _) { ns_value }
          ast::def_obj_field(_, _) { ns_value }
          ast::def_mod(_) { ns_module }
          ast::def_native_mod(_) { ns_module }
          ast::def_const(_) { ns_value }
          ast::def_arg(_, _) { ns_value }
          ast::def_local(_, _) { ns_value }
          ast::def_upvar(_, _, _) { ns_value }
          ast::def_variant(_, _) { ns_value }
          ast::def_ty(_) { ns_type }
          ast::def_binding(_) { ns_type }
          ast::def_use(_) { ns_module }
          ast::def_native_ty(_) { ns_type }
          ast::def_native_fn(_, _) { ns_value }
          ast::def_self(_) { ns_value }
        };
}

fn lookup_external(e: env, cnum: int, ids: [ident], ns: namespace) ->
   option::t<def> {
    for d: def in csearch::lookup_defs(e.sess.get_cstore(), cnum, ids) {
        e.ext_map.insert(def_id_of_def(d), ids);
        if ns == ns_for_def(d) { ret some(d); }
    }
    ret none::<def>;
}


// Collision detection
fn check_for_collisions(e: @env, c: ast::crate) {
    // Module indices make checking those relatively simple -- just check each
    // name for multiple entities in the same namespace.
    e.mod_map.values {|val|
        val.index.items {|k, v| check_mod_name(*e, k, v); };
    };
    // Other scopes have to be checked the hard way.
    let v =
        @{visit_item: bind check_item(e, _, _, _),
          visit_block: bind check_block(e, _, _, _),
          visit_arm: bind check_arm(e, _, _, _),
          visit_expr: bind check_expr(e, _, _, _),
          visit_ty: bind check_ty(e, _, _, _) with *visit::default_visitor()};
    visit::visit_crate(c, (), visit::mk_vt(v));
}

fn check_mod_name(e: env, name: ident, entries: list<mod_index_entry>) {
    let saw_mod = false;
    let saw_type = false;
    let saw_value = false;
    let entries = entries;
    fn dup(e: env, sp: span, word: str, name: ident) {
        e.sess.span_fatal(sp, "duplicate definition of " + word + name);
    }
    while true {
        alt entries {
          cons(entry, rest) {
            if !is_none(lookup_in_mie(e, entry, ns_value)) {
                if saw_value {
                    dup(e, mie_span(entry), "", name);
                } else { saw_value = true; }
            }
            if !is_none(lookup_in_mie(e, entry, ns_type)) {
                if saw_type {
                    dup(e, mie_span(entry), "type ", name);
                } else { saw_type = true; }
            }
            if !is_none(lookup_in_mie(e, entry, ns_module)) {
                if saw_mod {
                    dup(e, mie_span(entry), "module ", name);
                } else { saw_mod = true; }
            }
            entries = *rest;
          }
          nil. { break; }
        }
    }
}

fn mie_span(mie: mod_index_entry) -> span {
    ret alt mie {
          mie_view_item(item) { item.span }
          mie_import_ident(_, span) { span }
          mie_item(item) { item.span }
          mie_tag_variant(item, _) { item.span }
          mie_native_item(item) { item.span }
        };
}

fn check_item(e: @env, i: @ast::item, &&x: (), v: vt<()>) {
    fn typaram_names(tps: [ast::ty_param]) -> [ident] {
        let x: [ast::ident] = [];
        for tp: ast::ty_param in tps { x += [tp.ident]; }
        ret x;
    }
    visit::visit_item(i, x, v);
    alt i.node {
      ast::item_fn(f, ty_params) {
        check_fn(*e, i.span, f);
        ensure_unique(*e, i.span, typaram_names(ty_params), ident_id,
                      "type parameter");
      }
      ast::item_obj(ob, ty_params, _) {
        fn field_name(field: ast::obj_field) -> ident { ret field.ident; }
        ensure_unique(*e, i.span, ob.fields, field_name, "object field");
        for m: @ast::method in ob.methods {
            check_fn(*e, m.span, m.node.meth);
        }
        ensure_unique(*e, i.span, typaram_names(ty_params), ident_id,
                      "type parameter");
      }
      ast::item_tag(_, ty_params) {
        ensure_unique(*e, i.span, typaram_names(ty_params), ident_id,
                      "type parameter");
      }
      _ { }
    }
}

fn check_pat(ch: checker, p: @ast::pat) {
    ast_util::pat_bindings(p) {|p|
        let ident = alt p.node { pat_bind(n, _) { n } };
        add_name(ch, p.span, ident);
    };
}

fn check_arm(e: @env, a: ast::arm, &&x: (), v: vt<()>) {
    visit::visit_arm(a, x, v);
    let ch0 = checker(*e, "binding");
    check_pat(ch0, a.pats[0]);
    let seen0 = ch0.seen;
    let i = vec::len(a.pats);
    while i > 1u {
        i -= 1u;
        let ch = checker(*e, "binding");
        check_pat(ch, a.pats[i]);

        // Ensure the bindings introduced in this pattern are the same as in
        // the first pattern.
        if vec::len(ch.seen) != vec::len(seen0) {
            e.sess.span_err(a.pats[i].span,
                            "inconsistent number of bindings");
        } else {
            for name: ident in ch.seen {
                if is_none(vec::find(bind str::eq(name, _), seen0)) {
                    // Fight the alias checker
                    let name_ = name;
                    e.sess.span_err(a.pats[i].span,
                                    "binding " + name_ +
                                        " does not occur in first pattern");
                }
            }
        }
    }
}

fn check_block(e: @env, b: ast::blk, &&x: (), v: vt<()>) {
    visit::visit_block(b, x, v);
    let values = checker(*e, "value");
    let types = checker(*e, "type");
    let mods = checker(*e, "module");
    for st: @ast::stmt in b.node.stmts {
        alt st.node {
          ast::stmt_decl(d, _) {
            alt d.node {
              ast::decl_local(locs) {
                let local_values = checker(*e, "value");
                for (_, loc) in locs {
                    ast_util::pat_bindings(loc.node.pat) {|p|
                        let ident = alt p.node { pat_bind(n, _) { n } };
                        add_name(local_values, p.span, ident);
                        check_name(values, p.span, ident);
                    };
                }
              }
              ast::decl_item(it) {
                alt it.node {
                  ast::item_tag(variants, _) {
                    add_name(types, it.span, it.ident);
                    for v: ast::variant in variants {
                        add_name(values, v.span, v.node.name);
                    }
                  }
                  ast::item_mod(_) | ast::item_native_mod(_) {
                    add_name(mods, it.span, it.ident);
                  }
                  ast::item_const(_, _) | ast::item_fn(_, _) {
                    add_name(values, it.span, it.ident);
                  }
                  ast::item_ty(_, _) { add_name(types, it.span, it.ident); }
                  ast::item_res(_, _, _, _) | ast::item_obj(_, _, _) {
                    add_name(types, it.span, it.ident);
                    add_name(values, it.span, it.ident);
                  }
                  _ { }
                }
              }
            }
          }
          _ { }
        }
    }
}

fn check_fn(e: env, sp: span, f: ast::_fn) {
    fn arg_name(a: ast::arg) -> ident { ret a.ident; }
    ensure_unique(e, sp, f.decl.inputs, arg_name, "argument");
}

fn check_expr(e: @env, ex: @ast::expr, &&x: (), v: vt<()>) {
    alt ex.node {
      ast::expr_rec(fields, _) {
        fn field_name(f: ast::field) -> ident { ret f.node.ident; }
        ensure_unique(*e, ex.span, fields, field_name, "field");
      }
      _ { }
    }
    visit::visit_expr(ex, x, v);
}

fn check_ty(e: @env, ty: @ast::ty, &&x: (), v: vt<()>) {
    alt ty.node {
      ast::ty_rec(fields) {
        fn field_name(f: ast::ty_field) -> ident { ret f.node.ident; }
        ensure_unique(*e, ty.span, fields, field_name, "field");
      }
      _ { }
    }
    visit::visit_ty(ty, x, v);
}

type checker = @{mutable seen: [ident], kind: str, sess: session};

fn checker(e: env, kind: str) -> checker {
    let seen: [ident] = [];
    ret @{mutable seen: seen, kind: kind, sess: e.sess};
}

fn check_name(ch: checker, sp: span, name: ident) {
    for s: ident in ch.seen {
        if str::eq(s, name) {
            ch.sess.span_fatal(sp, "duplicate " + ch.kind + " name: " + name);
        }
    }
}
fn add_name(ch: checker, sp: span, name: ident) {
    check_name(ch, sp, name);
    ch.seen += [name];
}

fn ident_id(&&i: ident) -> ident { ret i; }

fn ensure_unique<T>(e: env, sp: span, elts: [T], id: fn(T) -> ident,
                    kind: str) {
    let ch = checker(e, kind);
    for elt: T in elts { add_name(ch, sp, id(elt)); }
}

fn check_exports(e: @env) {
    fn lookup_glob_any(e: @env, info: @indexed_mod, sp: span, path: str,
                       ident: ident) -> bool {
        let lookup =
            bind lookup_glob_in_mod(*e, info, sp, ident, _, inside);
        let (m, v, t) = (lookup(ns_module),
                         lookup(ns_value),
                         lookup(ns_type));
        maybe_add_reexport(e, path + ident, m);
        maybe_add_reexport(e, path + ident, v);
        maybe_add_reexport(e, path + ident, t);
        ret is_some(m) || is_some(v) || is_some(t);
    }

    fn maybe_add_reexport(e: @env, path: str, def: option::t<def>) {
        if option::is_some(def) {
            e.exp_map.insert(path, option::get(def));
        }
    }

    fn check_export(e: @env, ident: str, val: @indexed_mod, vi: @view_item) {
        if val.index.contains_key(ident) {
            let xs = val.index.get(ident);
            list::iter(xs) {|x|
                alt x {
                  mie_import_ident(id, _) {
                    alt e.imports.get(id) {
                      resolved(v, t, m, _, rid, _) {
                        maybe_add_reexport(e, val.path + rid, v);
                        maybe_add_reexport(e, val.path + rid, t);
                        maybe_add_reexport(e, val.path + rid, m);
                      }
                      _ { }
                    }
                  }
                  _ { }
                }
            }
        } else if lookup_glob_any(e, val, vi.span, val.path, ident) {
            // do nothing
        } else {
            e.sess.span_warn(vi.span,
                             #fmt("exported item %s is not defined", ident));
        }
    }

    e.mod_map.values {|val|
        alt val.m {
          some(m) {
            for vi in m.view_items {
                alt vi.node {
                  ast::view_item_export(idents, _) {
                    for ident in idents {
                        check_export(e, ident, val, vi);
                    }
                  }
                  _ { }
                }
            }
          }
          none. { }
        }
    };
}

// Impl resolution

fn resolve_impls(e: @env, c: @ast::crate) {
    visit::visit_crate(*c, nil, visit::mk_vt(@{
        visit_block: bind visit_block_with_impl_scope(e, _, _, _),
        visit_mod: bind visit_mod_with_impl_scope(e, _, _, _, _),
        visit_expr: bind resolve_impl_in_expr(e, _, _, _)
        with *visit::default_visitor()
    }));
}

fn find_impls_in_view_item(e: env, vi: @ast::view_item,
                           &impls: [@ast::item]) {
    alt vi.node {
      ast::view_item_import(ident, _, id) {
        // FIXME if single name, simply look in our own iscope
        alt e.imports.get(id) {
          resolved(_, _, _, is, _, _) { impls += *is; }
        }
      }
      ast::view_item_import_from(base, names, _) {
        for nm in names {
            alt e.imports.get(nm.node.id) {
              resolved(_, _, _, is, _, _) { impls += *is; }
            }
        }
      }
      ast::view_item_import_glob(ids, id) {
          alt e.imports.get(id) {
            is_glob(path, sc, sp) {
              alt follow_import(e, sc, *path, sp) {
                some(def) { find_impls_in_mod(e, def, impls, none); }
                _ {}
              }
            }
          }
      }
      _ {}
    }
}

fn find_impls_in_item(i: @ast::item, &impls: [@ast::item],
                      name: option::t<ident>, _dir: dir) {
    // FIXME check exports
    alt i.node {
      ast::item_impl(_, _, _) {
        if alt name { some(n) { n == i.ident } _ { true } } {
            impls += [i];
        }
      }
      _ {}
    }
}

fn find_impls_in_mod(e: env, m: def, &impls: [@ast::item],
                     name: option::t<ident>) {
    alt m {
      ast::def_mod(defid) {
        // FIXME external importing of impls
        if defid.crate == ast::local_crate {
            for i in option::get(e.mod_map.get(defid.node).m).items {
                find_impls_in_item(i, impls, name, outside);
            }
        }
      }
      _ {}
    }
}

type iscopes = list<@[@ast::item]>;

fn visit_block_with_impl_scope(e: @env, b: ast::blk, sc: iscopes,
                               v: vt<iscopes>) {
    let impls = [];
    for vi in b.node.view_items { find_impls_in_view_item(*e, vi, impls); }
    for st in b.node.stmts {
        alt st.node {
          ast::stmt_decl(@{node: ast::decl_item(i), _}, _) {
            find_impls_in_item(i, impls, none, inside);
          }
          _ {}
        }
    }
    let sc = vec::len(impls) > 0u ? cons(@impls, @sc) : sc;
    visit::visit_block(b, sc, v);
}

fn visit_mod_with_impl_scope(e: @env, m: ast::_mod, s: span, sc: iscopes,
                             v: vt<iscopes>) {
    let impls = [];
    for vi in m.view_items { find_impls_in_view_item(*e, vi, impls); }
    for i in m.items { find_impls_in_item(i, impls, none, inside); }
    visit::visit_mod(m, s, vec::len(impls) > 0u ? cons(@impls, @sc) : sc, v);
}

fn resolve_impl_in_expr(e: @env, x: @ast::expr, sc: iscopes, v: vt<iscopes>) {
    alt x.node {
      ast::expr_field(_, _) { e.impl_map.insert(x.id, sc); }
      _ {}
    }
    visit::visit_expr(x, sc, v);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
