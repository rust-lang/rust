import syntax::{ast, ast_util, codemap, ast_map};
import syntax::ast::*;
import ast::{ident, fn_ident, def, def_id, node_id};
import syntax::ast_util::{local_def, def_id_of_def, new_def_hash,
                          class_item_ident, path_to_ident};
import pat_util::*;

import syntax::attr;
import metadata::{csearch, cstore};
import driver::session::session;
import util::common::*;
import std::map::{int_hash, str_hash, hashmap};
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

enum scope {
    scope_toplevel,
    scope_crate,
    scope_item(@ast::item),
    scope_bare_fn(ast::fn_decl, node_id, [ast::ty_param]),
    scope_fn_expr(ast::fn_decl, node_id, [ast::ty_param]),
    scope_native_item(@ast::native_item),
    scope_loop(@ast::local), // there's only 1 decl per loop.
    scope_block(ast::blk, @mut uint, @mut uint),
    scope_arm(ast::arm),
    scope_method(node_id, [ast::ty_param]),
}

type scopes = @list<scope>;

fn top_scope() -> scopes {
    @cons(scope_crate, @cons(scope_toplevel, @nil))
}

enum import_state {
    todo(ast::ident, @[ast::ident], span, scopes),
    is_glob(@[ast::ident], scopes, span),
    resolving(span),
    resolved(option<def>, /* value */
             option<def>, /* type */
             option<def>, /* module */
             @[@_impl], /* impls */
             /* used for reporting unused import warning */
             ast::ident, span),
}

enum glob_import_state {
    glob_resolving(span),
    glob_resolved(option<def>,  /* value */
                  option<def>,  /* type */
                  option<def>), /* module */
}

type ext_hash = hashmap<{did: def_id, ident: str, ns: namespace}, def>;

fn new_ext_hash() -> ext_hash {
    type key = {did: def_id, ident: str, ns: namespace};
    fn hash(v: key) -> uint {
        str::hash(v.ident) + ast_util::hash_def(v.did) + v.ns as uint
    }
    fn eq(v1: key, v2: key) -> bool {
        ret ast_util::def_eq(v1.did, v2.did) &&
            str::eq(v1.ident, v2.ident) && v1.ns == v2.ns;
    }
    std::map::hashmap(hash, {|a, b| a == b})
}

enum mod_index_entry {
    mie_view_item(ident, node_id, span),
    mie_import_ident(node_id, span),
    mie_item(@ast::item),
    mie_native_item(@ast::native_item),
    mie_enum_variant(/* variant index */uint,
                     /*parts of enum item*/ [variant],
                    node_id, span),
}

type mod_index = hashmap<ident, @list<mod_index_entry>>;

// A tuple of an imported def and the view_path from its originating import
type glob_imp_def = {def: def, path: @ast::view_path};

type indexed_mod = {
    m: option<ast::_mod>,
    index: mod_index,
    mut glob_imports: [glob_imp_def],
    mut globbed_exports: [ident],
    glob_imported_names: hashmap<str, glob_import_state>,
    path: str
};

/* native modules can't contain enums, and we don't store their ASTs because
   we only need to look at them to determine exports, which they can't
   control.*/

type def_map = hashmap<node_id, def>;
type ext_map = hashmap<def_id, [ident]>;
type impl_map = hashmap<node_id, iscopes>;
type impl_cache = hashmap<def_id, option<@[@_impl]>>;

type exp = {reexp: bool, id: def_id};
type exp_map = hashmap<node_id, [exp]>;

type env =
    {cstore: cstore::cstore,
     def_map: def_map,
     ast_map: ast_map::map,
     imports: hashmap<node_id, import_state>,
     mut exp_map: exp_map,
     mod_map: hashmap<node_id, @indexed_mod>,
     block_map: hashmap<node_id, [glob_imp_def]>,
     ext_map: ext_map,
     impl_map: impl_map,
     impl_cache: impl_cache,
     ext_cache: ext_hash,
     used_imports: {mut track: bool,
                    mut data: [node_id]},
     mut reported: [{ident: str, sc: scope}],
     mut ignored_imports: [node_id],
     mut current_tp: option<uint>,
     mut resolve_unexported: bool,
     sess: session};


// Used to distinguish between lookups from outside and from inside modules,
// since export restrictions should only be applied for the former.
enum dir { inside, outside, }

enum namespace { ns_val, ns_type, ns_module, }

fn resolve_crate(sess: session, amap: ast_map::map, crate: @ast::crate) ->
   {def_map: def_map, exp_map: exp_map, impl_map: impl_map} {
    let e = create_env(sess, amap);
    map_crate(e, crate);
    resolve_imports(*e);
    check_exports(e);
    resolve_names(e, crate);
    resolve_impls(e, crate);
    // check_for_collisions must happen after resolve_names so we
    // don't complain if a pattern uses the same nullary enum twice
    check_for_collisions(e, *crate);

    // FIXME: move this to the lint pass when rewriting resolve.
    for sess.opts.lint_opts.each {|pair|
        let (lint,level) = pair;
        if lint == lint::unused_imports && level != lint::ignore {
            check_unused_imports(e, level);
            break;
        }
    }

    ret {def_map: e.def_map, exp_map: e.exp_map, impl_map: e.impl_map};
}

fn create_env(sess: session, amap: ast_map::map) -> @env {
    @{cstore: sess.cstore,
      def_map: int_hash(),
      ast_map: amap,
      imports: int_hash(),
      mut exp_map: int_hash(),
      mod_map: int_hash(),
      block_map: int_hash(),
      ext_map: new_def_hash(),
      impl_map: int_hash(),
      impl_cache: new_def_hash(),
      ext_cache: new_ext_hash(),
      used_imports: {mut track: false, mut data:  []},
      mut reported: [],
      mut ignored_imports: [],
      mut current_tp: none,
      mut resolve_unexported: false,
      sess: sess}
}

fn iter_export_paths(vi: ast::view_item, f: fn(vp: @ast::view_path)) {
    alt vi.node {
      ast::view_item_export(vps) {
        for vps.each {|vp|
            f(vp);
        }
      }
      _ {}
    }
}

fn iter_import_paths(vi: ast::view_item, f: fn(vp: @ast::view_path)) {
    alt vi.node {
      ast::view_item_import(vps) {
        for vps.each {|vp| f(vp);}
      }
      _ {}
    }
}

fn iter_effective_import_paths(vi: ast::view_item,
                               f: fn(vp: @ast::view_path)) {
    iter_import_paths(vi, f);
    iter_export_paths(vi) {|vp|
        alt vp.node {
          ast::view_path_simple(_, _, _) { }
          // FIXME: support uniform ident-list exports eventually;
          // at the moment they have half a meaning as reaching into
          // tags.
          ast::view_path_list(_, _, _) {}
          ast::view_path_glob(_,_) {
            f(vp);
          }
        }
    }
}

// Locate all modules and imports and index them, so that the next passes can
// resolve through them.
fn map_crate(e: @env, c: @ast::crate) {

    fn index_vi(e: @env, i: @ast::view_item, &&sc: scopes, _v: vt<scopes>) {
        iter_effective_import_paths(*i) { |vp|
            alt vp.node {
              ast::view_path_simple(name, path, id) {
                e.imports.insert(id, todo(name, @path.idents, vp.span,
                                          sc));
              }
              ast::view_path_glob(path, id) {
                e.imports.insert(id, is_glob(@path.idents, sc, vp.span));
              }
              ast::view_path_list(mod_path, idents, _) {
                for idents.each {|ident|
                    let t = todo(ident.node.name,
                                 @(mod_path.idents + [ident.node.name]),
                                 ident.span, sc);
                    e.imports.insert(ident.node.id, t);
                }
              }
            }
        }
    }

    fn path_from_scope(sc: scopes, n: str) -> str {
        let mut path = n + "::";
        list::iter(sc) {|s|
            alt s {
              scope_item(i) { path = i.ident + "::" + path; }
              _ {}
            }
        }
        path
    }

    fn index_i(e: @env, i: @ast::item, &&sc: scopes, v: vt<scopes>) {
        visit_item_with_scope(e, i, sc, v);
        alt i.node {
          ast::item_mod(md) {
            e.mod_map.insert(i.id,
                             @{m: some(md),
                               index: index_mod(md),
                               mut glob_imports: [],
                               mut globbed_exports: [],
                               glob_imported_names: str_hash(),
                               path: path_from_scope(sc, i.ident)});
          }
          ast::item_native_mod(nmd) {
            e.mod_map.insert(i.id,
                             @{m: none::<ast::_mod>,
                               index: index_nmod(nmd),
                               mut glob_imports: [],
                               mut globbed_exports: [],
                               glob_imported_names: str_hash(),
                               path: path_from_scope(sc, i.ident)});
          }
          _ { }
        }
    }

    // Note: a glob export works as an implicit import, along with a
    // re-export of anything that was exported at the glob-target location.
    // So we wind up reusing the glob-import machinery when looking at
    // glob exports. They just do re-exporting in a later step.
    fn link_glob(e: @env, vi: @ast::view_item, &&sc: scopes, _v: vt<scopes>) {
        iter_effective_import_paths(*vi) { |vp|
            alt vp.node {
              ast::view_path_glob(path, _) {
                alt follow_import(*e, sc, path.idents, vp.span) {
                  some(imp) {
                    let glob = {def: imp, path: vp};
                    alt list::head(sc) {
                      scope_item(i) {
                        e.mod_map.get(i.id).glob_imports += [glob];
                      }
                      scope_block(b, _, _) {
                        let globs = alt e.block_map.find(b.node.id) {
                          some(globs) { globs + [glob] }
                          none { [glob] }
                        };
                        e.block_map.insert(b.node.id, globs);
                      }
                      scope_crate {
                        e.mod_map.get(ast::crate_node_id).glob_imports
                            += [glob];
                      }
                      _ { e.sess.span_bug(vi.span, "unexpected scope in a \
                                                    glob import"); }
                    }
                  }
                  _ { }
                }
              }
              _ { }
            }
        }
    }

    // First, find all the modules, and index the names that they contain
    let v_map_mod =
        @{visit_view_item: bind index_vi(e, _, _, _),
          visit_item: bind index_i(e, _, _, _),
          visit_block: visit_block_with_scope
          with *visit::default_visitor::<scopes>()};
    visit::visit_crate(*c, top_scope(), visit::mk_vt(v_map_mod));

    // Register the top-level mod
    e.mod_map.insert(ast::crate_node_id,
                     @{m: some(c.node.module),
                       index: index_mod(c.node.module),
                       mut glob_imports: [],
                       mut globbed_exports: [],
                       glob_imported_names: str_hash(),
                       path: ""});

    // Next, assemble the links for globbed imports and exports.
    let v_link_glob =
        @{visit_view_item: bind link_glob(e, _, _, _),
          visit_block: visit_block_with_scope,
          visit_item: bind visit_item_with_scope(e, _, _, _)
          with *visit::default_visitor::<scopes>()};
    visit::visit_crate(*c, top_scope(), visit::mk_vt(v_link_glob));

}

fn resolve_imports(e: env) {
    e.used_imports.track = true;
    for e.imports.each {|id, v|
        alt check v {
          todo(name, path, span, scopes) {
            resolve_import(e, id, name, *path, span, scopes);
          }
          resolved(_, _, _, _, _, _) | is_glob(_, _, _) { }
        }
    }
    e.used_imports.track = false;
    e.sess.abort_if_errors();
}

// FIXME (#1634): move this to the lint pass when rewriting resolve. It's
// using lint-specific control flags presently but resolve-specific data
// structures. Should use the general lint framework (with scopes, attrs).
fn check_unused_imports(e: @env, level: lint::level) {
    for e.imports.each {|k, v|
        alt v {
            resolved(_, _, _, _, name, sp) {
              if !vec::contains(e.used_imports.data, k) {
                  alt level {
                    lint::warn {
                      e.sess.span_warn(sp, "unused import " + name);
                    }
                    lint::error {
                      e.sess.span_err(sp, "unused import " + name);
                    }
                    lint::ignore {
                    }
                  }
              }
            }
            _ { }
        }
    };
}

fn resolve_capture_item(e: @env, sc: scopes, cap_item: ast::capture_item) {
    let dcur = lookup_in_scope_strict(
        *e, sc, cap_item.span, cap_item.name, ns_val);
    maybe_insert(e, cap_item.id, dcur);
}

fn maybe_insert(e: @env, id: node_id, def: option<def>) {
    alt def {
       some(df) { e.def_map.insert(id, df); }
       _ {}
    }
}

fn resolve_iface_ref(p: @iface_ref, sc: scopes, e: @env) {
    maybe_insert(e, p.id,
       lookup_path_strict(*e, sc, p.path.span, p.path, ns_type));
}

fn resolve_names(e: @env, c: @ast::crate) {
    e.used_imports.track = true;
    let v =
        @{visit_native_item: visit_native_item_with_scope,
          visit_item: bind walk_item(e, _, _, _),
          visit_block: visit_block_with_scope,
          visit_decl: visit_decl_with_scope,
          visit_arm: visit_arm_with_scope,
          visit_local: bind visit_local_with_scope(e, _, _, _),
          visit_pat: bind walk_pat(e, _, _, _),
          visit_expr: bind walk_expr(e, _, _, _),
          visit_ty: bind walk_ty(e, _, _, _),
          visit_ty_params: bind walk_tps(e, _, _, _),
          visit_constr: bind walk_constr(e, _, _, _, _, _),
          visit_fn: bind visit_fn_with_scope(e, _, _, _, _, _, _, _)
          with *visit::default_visitor()};
    visit::visit_crate(*c, top_scope(), visit::mk_vt(v));
    e.used_imports.track = false;
    e.sess.abort_if_errors();

    fn walk_item(e: @env, i: @ast::item, &&sc: scopes, v: vt<scopes>) {
        visit_item_with_scope(e, i, sc, v);
        alt i.node {
          /* At this point, the code knows what ifaces the iface refs
             refer to, so it's possible to resolve them.
           */
          ast::item_impl(_, _, ifce, _, _) {
            ifce.iter {|p| resolve_iface_ref(p, sc, e);}
          }
          ast::item_class(_, ifaces, _, _, _, _) {
            for ifaces.each {|p|
               resolve_iface_ref(p, sc, e);
            }
          }
          _ {}
        }
    }

    fn walk_expr(e: @env, exp: @ast::expr, &&sc: scopes, v: vt<scopes>) {
        visit::visit_expr(exp, sc, v);
        alt exp.node {
          ast::expr_path(p) {
            maybe_insert(e, exp.id,
                         lookup_path_strict(*e, sc, exp.span, p, ns_val));
          }
          ast::expr_fn(_, _, _, cap_clause) |
          ast::expr_fn_block(_, _, cap_clause) {
            for (*cap_clause).each { |ci|
                resolve_capture_item(e, sc, ci);
            }
          }
          _ { }
        }
    }
    fn walk_ty(e: @env, t: @ast::ty, &&sc: scopes, v: vt<scopes>) {
        visit::visit_ty(t, sc, v);
        alt t.node {
          ast::ty_path(p, id) {
            maybe_insert(e, id,
                         lookup_path_strict(*e, sc, t.span, p, ns_type));
          }
          _ { }
        }
    }
    fn walk_tps(e: @env, tps: [ast::ty_param], &&sc: scopes, v: vt<scopes>) {
        let outer_current_tp = e.current_tp;
        let mut current = 0u;
        for tps.each {|tp|
            e.current_tp = some(current);
            for vec::each(*tp.bounds) {|bound|
                alt bound {
                  bound_iface(t) { v.visit_ty(t, sc, v); }
                  _ {}
                }
            }
            current += 1u;
        }
        e.current_tp = outer_current_tp;
    }
    fn walk_constr(e: @env, p: @ast::path, sp: span, id: node_id,
                   &&sc: scopes, _v: vt<scopes>) {
        maybe_insert(e, id, lookup_path_strict(*e, sc, sp, p, ns_val));
    }
    fn walk_pat(e: @env, pat: @ast::pat, &&sc: scopes, v: vt<scopes>) {
        visit::visit_pat(pat, sc, v);
        alt pat.node {
          ast::pat_enum(p, _) {
            alt lookup_path_strict(*e, sc, p.span, p, ns_val) {
              some(fnd@ast::def_variant(_,_)) {
                e.def_map.insert(pat.id, fnd);
              }
              _ {
                e.sess.span_err(p.span,
                                "not an enum variant: " +
                                    ast_util::path_name(p));
              }
            }
          }
          /* Here we determine whether a given pat_ident binds a new
           variable or refers to a nullary enum. */
          ast::pat_ident(p, none) {
              alt lookup_in_scope(*e, sc, p.span, path_to_ident(p),
                                  ns_val, false) {
                some(fnd@ast::def_variant(_,_)) {
                    e.def_map.insert(pat.id, fnd);
                }
                some(fnd@ast::def_const(_)) {
                    e.sess.span_err(p.span, "pattern variable conflicts \
                       with constant '" + path_to_ident(p) + "'");
                }
                // Binds a var -- nothing needs to be done
                _ {}
              }
          }
          _ { }
        }
    }
}


// Visit helper functions
/*
  This is used in more than one context, thus should only call generic
  visit methods. Called both from map_crate and resolve_names.
 */
fn visit_item_with_scope(e: @env, i: @ast::item,
                         &&sc: scopes, v: vt<scopes>) {
    // Some magic here. Items with the !resolve_unexported attribute
    // cause us to consider every name to be exported when resolving their
    // contents. This is used to allow the test runner to run unexported
    // tests.
    let old_resolve_unexported = e.resolve_unexported;
    e.resolve_unexported |=
        attr::contains_name(attr::attr_metas(i.attrs),
                            "!resolve_unexported");

    let sc = @cons(scope_item(i), sc);
    alt i.node {
      ast::item_impl(tps, _, ifce, sty, methods) {
        visit::visit_ty_params(tps, sc, v);
        option::iter(ifce) {|p| visit::visit_path(p.path, sc, v)};
        v.visit_ty(sty, sc, v);
        for methods.each {|m|
            v.visit_ty_params(m.tps, sc, v);
            let msc = @cons(scope_method(m.self_id, tps + m.tps), sc);
            v.visit_fn(visit::fk_method(m.ident, [], m),
                       m.decl, m.body, m.span, m.id, msc, v);
        }
      }
      ast::item_iface(tps, _, methods) {
        visit::visit_ty_params(tps, sc, v);
        for methods.each {|m|
            let msc = @cons(scope_method(i.id, tps + m.tps), sc);
            for m.decl.inputs.each {|a| v.visit_ty(a.ty, msc, v); }
            v.visit_ty(m.decl.output, msc, v);
        }
      }
      ast::item_class(tps, ifaces, members, ctor, m_dtor, _) {
        visit::visit_ty_params(tps, sc, v);
        // Can maybe skip this now that we require self on class fields
        let class_scope = @cons(scope_item(i), sc);
        /* visit the constructor... */
        let ctor_scope = @cons(scope_method(ctor.node.self_id, tps),
                               class_scope);
        /* visit the iface refs in the class scope */
        for ifaces.each {|p|
            visit::visit_path(p.path, class_scope, v);
        }
        // FIXME: should be fk_ctor?
        visit_fn_with_scope(e, visit::fk_item_fn(i.ident, tps), ctor.node.dec,
                            ctor.node.body, ctor.span, ctor.node.id,
                            ctor_scope, v);
        option::iter(m_dtor) {|dtor|
          let dtor_scope = @cons(scope_method(dtor.node.self_id, tps),
                                 class_scope);

          visit_fn_with_scope(e, visit::fk_dtor(tps, dtor.node.self_id,
                                                local_def(i.id)),
                            ast_util::dtor_dec(),
                            dtor.node.body, dtor.span, dtor.node.id,
                            dtor_scope, v);
        };
        /* visit the items */
        for members.each {|cm|
            alt cm.node {
              class_method(m) {
                  let msc = @cons(scope_method(m.self_id, tps + m.tps),
                                  class_scope);
                  visit_fn_with_scope(e,
                     visit::fk_item_fn(m.ident, tps), m.decl, m.body,
                                 m.span, m.id, msc, v); }
              instance_var(_,t,_,_,_) { v.visit_ty(t, class_scope, v); }
            }
        }
      }
      _ { visit::visit_item(i, sc, v); }
    }

    e.resolve_unexported = old_resolve_unexported;
}

fn visit_native_item_with_scope(ni: @ast::native_item, &&sc: scopes,
                                v: vt<scopes>) {
    visit::visit_native_item(ni, @cons(scope_native_item(ni), sc), v);
}

fn visit_fn_with_scope(e: @env, fk: visit::fn_kind, decl: ast::fn_decl,
                       body: ast::blk, sp: span,
                       id: node_id, &&sc: scopes, v: vt<scopes>) {
    // is this a main fn declaration?
    alt fk {
      visit::fk_item_fn(nm, _) {
        if is_main_name([ast_map::path_name(nm)]) &&
           !e.sess.building_library {
            // This is a main function -- set it in the session
            // as the main ID
            e.sess.main_fn = some((id, sp));
        }
      }
      _ { /* fallthrough */ }
    }

    // here's where we need to set up the mapping
    // for f's constrs in the table.
    for decl.constraints.each {|c| resolve_constr(e, c, sc, v); }
    let scope = alt fk {
      visit::fk_item_fn(_, tps) | visit::fk_res(_, tps, _) |
      visit::fk_method(_, tps, _) | visit::fk_ctor(_, tps, _, _)  |
      visit::fk_dtor(tps, _, _) {
        scope_bare_fn(decl, id, tps) }
      visit::fk_anon(ast::proto_bare, _) {
        scope_bare_fn(decl, id, []) }
      visit::fk_anon(_, _) | visit::fk_fn_block(_) {
        scope_fn_expr(decl, id, []) }
    };

    visit::visit_fn(fk, decl, body, sp, id, @cons(scope, sc), v);
}

fn visit_block_with_scope(b: ast::blk, &&sc: scopes, v: vt<scopes>) {
    let pos = @mut 0u, loc = @mut 0u;
    let block_sc = @cons(scope_block(b, pos, loc), sc);
    for b.node.view_items.each {|vi| v.visit_view_item(vi, block_sc, v); }
    for b.node.stmts.each {|stmt|
        v.visit_stmt(stmt, block_sc, v);;
        *pos += 1u;;
        *loc = 0u;
    }
    visit::visit_expr_opt(b.node.expr, block_sc, v);
}

fn visit_decl_with_scope(d: @decl, &&sc: scopes, v: vt<scopes>) {
    let loc_pos = alt list::head(sc) {
      scope_block(_, _, pos) { pos }
      _ { @mut 0u }
    };
    alt d.node {
      decl_local(locs) {
        for locs.each {|loc| v.visit_local(loc, sc, v);; *loc_pos += 1u; }
      }
      decl_item(it) { v.visit_item(it, sc, v); }
    }
}

fn visit_arm_with_scope(a: ast::arm, &&sc: scopes, v: vt<scopes>) {
    for a.pats.each {|p| v.visit_pat(p, sc, v); }
    let sc_inner = @cons(scope_arm(a), sc);
    visit::visit_expr_opt(a.guard, sc_inner, v);
    v.visit_block(a.body, sc_inner, v);
}

// This is only for irrefutable patterns (e.g. ones that appear in a let)
// So if x occurs, and x is already known to be a enum, that's always an error
fn visit_local_with_scope(e: @env, loc: @local, &&sc: scopes, v:vt<scopes>) {
    // Check whether the given local has the same name as a enum that's in
    // scope. We disallow this, in order to make alt patterns consisting of a
    // single identifier unambiguous (does the pattern "foo" refer to enum
    // foo, or is it binding a new name foo?)
    ast_util::walk_pat(loc.node.pat) { |p|
        alt p.node {
          pat_ident(path, _) {
            alt lookup_in_scope(*e, sc, loc.span, path_to_ident(path),
                                ns_val, false) {
              some(ast::def_variant(enum_id, variant_id)) {
                // Declaration shadows an enum that's in scope.
                // That's an error.
                e.sess.span_err(loc.span,
                                #fmt("declaration of `%s` shadows an \
                                      enum that's in scope",
                                     path_to_ident(path)));
              }
              _ {}
            }
          }
          _ {}
        }
    }
    visit::visit_local(loc, sc, v);
}


fn follow_import(e: env, &&sc: scopes, path: [ident], sp: span) ->
   option<def> {
    let path_len = vec::len(path);
    let mut dcur = lookup_in_scope_strict(e, sc, sp, path[0], ns_module);
    let mut i = 1u;
    loop {
       alt copy dcur {
          some(dcur_def) {
            if i == path_len { break; }
            dcur =
                lookup_in_mod_strict(e, dcur_def, sp, path[i],
                                 ns_module, outside);
            i += 1u;
          }
          none { break; }
       }
    }
    if i == path_len {
       alt dcur {
          some(ast::def_mod(_)) | some(ast::def_native_mod(_)) { ret dcur; }
          _ {
            e.sess.span_err(sp, str::connect(path, "::") +
                            " does not name a module.");
            ret none;
          }
        }
    } else { ret none; }
}

fn resolve_constr(e: @env, c: @ast::constr, &&sc: scopes, _v: vt<scopes>) {
    alt lookup_path_strict(*e, sc, c.span, c.node.path, ns_val) {
       some(d@ast::def_fn(_,ast::pure_fn)) {
         e.def_map.insert(c.node.id, d);
       }
       _ {
           let s = path_to_str(c.node.path);
           e.sess.span_err(c.span, #fmt("%s is not declared pure. Try \
             `pure fn %s` instead of `fn %s`.", s, s, s));
       }
    }
}

// Import resolution
fn resolve_import(e: env, n_id: node_id, name: ast::ident,
                  ids: [ast::ident], sp: codemap::span, &&sc: scopes) {
    fn register(e: env, id: node_id, cx: ctxt, sp: codemap::span,
                name: ast::ident, lookup: fn(namespace) -> option<def>,
                impls: [@_impl]) {
        let val = lookup(ns_val), typ = lookup(ns_type),
            md = lookup(ns_module);
        if is_none(val) && is_none(typ) && is_none(md) &&
           vec::len(impls) == 0u {
            unresolved_err(e, cx, sp, name, "import");
        } else {
            e.imports.insert(id, resolved(val, typ, md, @impls, name, sp));
        }
    }
    // Temporarily disable this import and the imports coming after during
    // resolution of this import.
    fn find_imports_after(e: env, id: node_id, &&sc: scopes) -> [node_id] {
        fn lst(my_id: node_id, vis: [@view_item]) -> [node_id] {
            let mut imports = [], found = false;
            for vis.each {|vi|
                iter_effective_import_paths(*vi) {|vp|
                    alt vp.node {
                      view_path_simple(_, _, id)
                      | view_path_glob(_, id) {
                        if id == my_id { found = true; }
                        if found { imports += [id]; }
                      }
                      view_path_list(_, ids, _) {
                        for ids.each {|id|
                            if id.node.id == my_id { found = true; }
                            if found { imports += [id.node.id]; }
                        }
                      }
                    }
                }
            }
            imports
        }
        alt *sc {
          cons(scope_item(@{node: item_mod(m), _}), _) {
            lst(id, m.view_items)
          }
          cons(scope_item(@{node: item_native_mod(m), _}), _) {
            lst(id, m.view_items)
          }
          cons(scope_block(b, _, _), _) {
            lst(id, b.node.view_items)
          }
          cons(scope_crate, _) {
            lst(id,
                option::get(e.mod_map.get(ast::crate_node_id).m).view_items)
          }
          _ {
              e.sess.bug("find_imports_after: nil or unexpected scope");
          }
        }
    }
    // This function has cleanup code at the end. Do not return without going
    // through that.
    e.imports.insert(n_id, resolving(sp));
    let mut ignored = find_imports_after(e, n_id, sc);
    e.ignored_imports <-> ignored;
    let n_idents = vec::len(ids);
    let end_id = ids[n_idents - 1u];
    if n_idents == 1u {
        register(e, n_id, in_scope(sc), sp, name,
                 {|ns| lookup_in_scope(e, sc, sp, end_id, ns, true) }, []);
    } else {
        alt lookup_in_scope(e, sc, sp, ids[0], ns_module, true) {
          none {
            unresolved_err(e, in_scope(sc), sp, ids[0], ns_name(ns_module));
          }
          some(dcur_) {
            let mut dcur = dcur_, i = 1u;
            loop {
                if i == n_idents - 1u {
                    let mut impls = [];
                    find_impls_in_mod(e, dcur, impls, some(end_id));
                    register(e, n_id, in_mod(dcur), sp, name, {|ns|
                        lookup_in_mod(e, dcur, sp, end_id, ns, outside)
                    }, impls);
                    break;
                } else {
                    dcur = alt lookup_in_mod(e, dcur, sp, ids[i], ns_module,
                                             outside) {
                      some(dcur) { dcur }
                      none {
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
    alt e.imports.find(n_id) {
      some(resolving(sp)) {
        e.imports.insert(n_id, resolved(none, none, none, @[], "", sp));
      }
      _ { }
    }
}


// Utilities
fn ns_name(ns: namespace) -> str {
    alt ns {
      ns_type { "typename" }
      ns_val { "name" }
      ns_module { "modulename" }
    }
}

enum ctxt { in_mod(def), in_scope(scopes), }

fn unresolved_err(e: env, cx: ctxt, sp: span, name: ident, kind: str) {
    fn find_fn_or_mod_scope(sc: scopes) -> option<scope> {
        for list::each(sc) {|cur|
            alt cur {
              scope_crate | scope_bare_fn(_, _, _) | scope_fn_expr(_, _, _) |
              scope_item(@{node: ast::item_mod(_), _}) {
                ret some(cur);
              }
              _ {}
            }
        }
        ret none;
    }
    let mut path = name;
    alt cx {
      in_scope(sc) {
        alt find_fn_or_mod_scope(sc) {
          some(err_scope) {
            for e.reported.each {|rs|
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
        } else if did.node != ast::crate_node_id {
            let paths = e.ext_map.get(did);
            path = str::connect(paths + [path], "::");
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
fn lookup_path_strict(e: env, &&sc: scopes, sp: span, pth: @ast::path,
                      ns: namespace) -> option<def> {
    let n_idents = vec::len(pth.idents);
    let headns = if n_idents == 1u { ns } else { ns_module };

    let first_scope = if pth.global { top_scope() } else { sc };

    let dcur_ =
        lookup_in_scope_strict(e, first_scope, sp, pth.idents[0], headns);

    alt dcur_ {
      none { ret none; }
      some(dcur__) {
         let mut i = 1u;
         let mut dcur = dcur__;
         while i < n_idents {
            let curns = if n_idents == i + 1u { ns } else { ns_module };
            alt lookup_in_mod_strict(e, dcur, sp, pth.idents[i],
                                 curns, outside) {
               none { break; }
               some(thing) { dcur = thing; }
            }
            i += 1u;
         }
         ret some(dcur);
      }
    }
}

fn lookup_in_scope_strict(e: env, &&sc: scopes, sp: span, name: ident,
                          ns: namespace) -> option<def> {
    alt lookup_in_scope(e, sc, sp, name, ns, true) {
      none {
        unresolved_err(e, in_scope(sc), sp, name, ns_name(ns));
        ret none;
      }
      some(d) { ret some(d); }
    }
}

fn scope_is_fn(sc: scope) -> bool {
    ret alt sc {
      scope_bare_fn(_, _, _) | scope_native_item(_) { true }
      _ { false }
    };
}

// Returns:
//   none - does not close
//   some(node_id) - closes via the expr w/ node_id
fn scope_closes(sc: scope) -> option<node_id> {
    alt sc {
      scope_fn_expr(_, node_id, _) { some(node_id) }
      _ { none }
    }
}

fn def_is_local(d: def) -> bool {
    alt d {
      ast::def_arg(_, _) | ast::def_local(_, _) | ast::def_binding(_) |
      ast::def_upvar(_, _, _) { true }
      _ { false }
    }
}

fn def_is_self(d: def) -> bool {
    alt d {
      ast::def_self(_) { true }
      _ { false }
    }
}

fn def_is_ty_arg(d: def) -> bool {
    ret alt d { ast::def_ty_param(_, _) { true } _ { false } };
}

fn lookup_in_scope(e: env, &&sc: scopes, sp: span, name: ident, ns: namespace,
                   check_capture: bool) -> option<def> {

    fn in_scope(e: env, sp: span, name: ident, s: scope, ns: namespace) ->
       option<def> {
        alt s {
          scope_toplevel {
            if ns == ns_type {
                ret some(ast::def_prim_ty(alt name {
                  "bool" { ast::ty_bool }
                  "int" { ast::ty_int(ast::ty_i) }
                  "uint" { ast::ty_uint(ast::ty_u) }
                  "float" { ast::ty_float(ast::ty_f) }
                  "str" { ast::ty_str }
                  "char" { ast::ty_int(ast::ty_char) }
                  "i8" { ast::ty_int(ast::ty_i8) }
                  "i16" { ast::ty_int(ast::ty_i16) }
                  "i32" { ast::ty_int(ast::ty_i32) }
                  "i64" { ast::ty_int(ast::ty_i64) }
                  "u8" { ast::ty_uint(ast::ty_u8) }
                  "u16" { ast::ty_uint(ast::ty_u16) }
                  "u32" { ast::ty_uint(ast::ty_u32) }
                  "u64" { ast::ty_uint(ast::ty_u64) }
                  "f32" { ast::ty_float(ast::ty_f32) }
                  "f64" { ast::ty_float(ast::ty_f64) }
                  _ { ret none; }
                }));
            }
          }
          scope_crate {
            ret lookup_in_local_mod(e, ast::crate_node_id, sp,
                                    name, ns, inside);
          }
          scope_item(it) {
            alt it.node {
              ast::item_impl(tps, _, _, _, _) {
                if ns == ns_type { ret lookup_in_ty_params(e, name, tps); }
              }
              ast::item_enum(_, tps, _) | ast::item_ty(_, tps, _) {
                if ns == ns_type { ret lookup_in_ty_params(e, name, tps); }
              }
              ast::item_iface(tps, _, _) {
                if ns == ns_type {
                    if name == "self" {
                        ret some(def_self(it.id));
                    }
                    ret lookup_in_ty_params(e, name, tps);
                }
              }
              ast::item_mod(_) {
                ret lookup_in_local_mod(e, it.id, sp, name, ns, inside);
              }
              ast::item_native_mod(m) {
                ret lookup_in_local_native_mod(e, it.id, sp, name, ns);
              }
              ast::item_class(tps, _, members, ctor, _, _) {
                  if ns == ns_type {
                    ret lookup_in_ty_params(e, name, tps);
                  }
                  if ns == ns_val && name == it.ident {
                      ret some(ast::def_fn(local_def(ctor.node.id),
                                           ast::impure_fn));
                  }
                  // FIXME: AST allows other items to appear in a class,
                  // but that might not be wise
              }
              _ { }
            }
          }
          scope_method(id, tps) {
            if (name == "self" && ns == ns_val) {
                ret some(ast::def_self(id));
            } else if ns == ns_type {
                ret lookup_in_ty_params(e, name, tps);
            }
          }
          scope_native_item(it) {
            alt check it.node {
              ast::native_item_fn(decl, ty_params) {
                ret lookup_in_fn(e, name, decl, ty_params, ns);
              }
            }
          }
          scope_bare_fn(decl, _, ty_params) |
          scope_fn_expr(decl, _, ty_params) {
            ret lookup_in_fn(e, name, decl, ty_params, ns);
          }
          scope_loop(local) {
            if ns == ns_val {
                alt lookup_in_pat(e, name, local.node.pat) {
                  some(nid) { ret some(ast::def_binding(nid)); }
                  _ { }
                }
            }
          }
          scope_block(b, pos, loc) {
            ret lookup_in_block(e, name, sp, b.node, *pos, *loc, ns);
          }
          scope_arm(a) {
            if ns == ns_val {
                alt lookup_in_pat(e, name, a.pats[0]) {
                  some(nid) { ret some(ast::def_binding(nid)); }
                  _ { ret none; }
                }
            }
          }
        }
        ret none;
    }
    let mut left_fn = false;
    let mut closing = [];
    // Used to determine whether self is in scope
    let mut left_fn_level2 = false;
    let mut sc = sc;
    loop {
        alt *sc {
          nil { ret none; }
          cons(hd, tl) {
              alt in_scope(e, sp, name, hd, ns) {
               some(df_) {
                 let mut df = df_;
                 let local = def_is_local(df), self_scope = def_is_self(df);
                 if check_capture &&
                     (left_fn && local || left_fn_level2 && self_scope
                      || scope_is_fn(hd) && left_fn && def_is_ty_arg(df)) {
                     let msg = if ns == ns_type {
                         "attempt to use a type argument out of scope"
                     } else {
                         "attempted dynamic environment-capture"
                     };
                     e.sess.span_fatal(sp, msg);
                } else if local || self_scope {
                    let mut i = vec::len(closing);
                    while i > 0u {
                        i -= 1u;
                        #debug["name=%s df=%?", name, df];
                        assert def_is_local(df) || def_is_self(df);
                        let df_id = def_id_of_def(df).node;
                        df = ast::def_upvar(df_id, @df, closing[i]);
                    }
                }
                ret some(df);
            }
            _ {}
        }
        if left_fn {
           left_fn_level2 = true;
        } else if ns != ns_module {
                left_fn = scope_is_fn(hd);
                alt scope_closes(hd) {
                  some(node_id) { closing += [node_id]; }
                  _ { }
                }
            }
            sc = tl;
        }
      }
    };
}

fn lookup_in_ty_params(e: env, name: ident, ty_params: [ast::ty_param])
    -> option<def> {
    let mut n = 0u;
    for ty_params.each {|tp|
        if str::eq(tp.ident, name) && alt e.current_tp {
            some(cur) { n < cur } none { true }
        } { ret some(ast::def_ty_param(local_def(tp.id), n)); }
        n += 1u;
    }
    ret none;
}

fn lookup_in_pat(e: env, name: ident, pat: @ast::pat) -> option<node_id> {
    let mut found = none;

    pat_util::pat_bindings(e.def_map, pat) {|p_id, _sp, n|
        if str::eq(path_to_ident(n), name)
                    { found = some(p_id); }
    };
    ret found;
}

fn lookup_in_fn(e: env, name: ident, decl: ast::fn_decl,
                ty_params: [ast::ty_param],
                ns: namespace) -> option<def> {
    alt ns {
      ns_val {
        for decl.inputs.each {|a|
            if str::eq(a.ident, name) {
                ret some(ast::def_arg(a.id, a.mode));
            }
        }
        ret none;
      }
      ns_type { ret lookup_in_ty_params(e, name, ty_params); }
      _ { ret none; }
    }
}

fn lookup_in_block(e: env, name: ident, sp: span, b: ast::blk_, pos: uint,
                   loc_pos: uint, ns: namespace) -> option<def> {

    let mut i = vec::len(b.stmts);
    while i > 0u {
        i -= 1u;
        let st = b.stmts[i];
        alt st.node {
          ast::stmt_decl(d, _) {
            alt d.node {
              ast::decl_local(locs) {
                if i <= pos {
                    let mut j = vec::len(locs);
                    while j > 0u {
                        j -= 1u;
                        let loc = locs[j];
                        if ns == ns_val && (i < pos || j < loc_pos) {
                            alt lookup_in_pat(e, name, loc.node.pat) {
                              some(nid) {
                                ret some(ast::def_local(nid,
                                                        loc.node.is_mutbl));
                              }
                              _ { }
                            }
                        }
                    }
                }
              }
              ast::decl_item(it) {
                alt it.node {
                  ast::item_enum(variants, _, _) {
                    if ns == ns_type {
                        if str::eq(it.ident, name) {
                            ret some(ast::def_ty(local_def(it.id)));
                        }
                    } else {
                        alt ns {
                           ns_val {
                               for variants.each {|v|
                                  if str::eq(v.node.name, name) {
                                     let i = v.node.id;
                                     ret some(ast::def_variant
                                        (local_def(it.id), local_def(i)));
                                  }
                               }
                          }
                           _ {}
                        }
                    }
                  }
                  _ {
                    if str::eq(it.ident, name) {
                        let found = found_def_item(it, ns);
                        if !is_none(found) {
                            ret found;
                        }
                    }
                  }
                }
              }
            }
          }
          _ { }
        }
    }
    for b.view_items.each {|vi|
        let mut is_import = false;
        alt vi.node {
          ast::view_item_import(_) { is_import = true; }
          _ {}
        }

        alt vi.node {

          ast::view_item_import(vps) | ast::view_item_export(vps) {
            for vps.each {|vp|
                alt vp.node {
                  ast::view_path_simple(ident, _, id) {
                    if is_import && name == ident {
                        ret lookup_import(e, id, ns);
                    }
                  }

                  ast::view_path_list(path, idents, _) {
                    for idents.each {|ident|
                        if name == ident.node.name {
                            ret lookup_import(e, ident.node.id, ns);
                        }
                    }
                  }

                  ast::view_path_glob(_, _) {
                    alt e.block_map.find(b.id) {
                      some(globs) {
                        let found = lookup_in_globs(e, globs, sp, name,
                                                    ns, inside);
                        if found != none {
                            ret found;
                        }
                      }
                      _ {}
                    }
                  }
                }
            }
          }
          _ { e.sess.span_bug(vi.span, "unexpected view_item in block"); }
        }
    }
    ret none;
}

fn found_def_item(i: @ast::item, ns: namespace) -> option<def> {
    alt i.node {
      ast::item_const(*) {
        if ns == ns_val {
            ret some(ast::def_const(local_def(i.id))); }
      }
      ast::item_fn(decl, _, _) {
        if ns == ns_val {
            ret some(ast::def_fn(local_def(i.id), decl.purity));
        }
      }
      ast::item_mod(_) {
        if ns == ns_module { ret some(ast::def_mod(local_def(i.id))); }
      }
      ast::item_native_mod(_) {
        if ns == ns_module { ret some(ast::def_native_mod(local_def(i.id))); }
      }
      ast::item_ty(*) | item_iface(*) | item_enum(*) {
        if ns == ns_type { ret some(ast::def_ty(local_def(i.id))); }
      }
      ast::item_res(_, _, _, _, ctor_id, _) {
        alt ns {
          ns_val {
            ret some(ast::def_fn(local_def(ctor_id), ast::impure_fn));
          }
          ns_type { ret some(ast::def_ty(local_def(i.id))); }
          _ { }
        }
      }
      ast::item_class(*) {
          if ns == ns_type {
            ret some(ast::def_class(local_def(i.id)));
          }
      }
      ast::item_impl(*) { /* ??? */ }
    }
    ret none;
}

fn lookup_in_mod_strict(e: env, m: def, sp: span, name: ident,
                        ns: namespace, dr: dir) -> option<def> {
    alt lookup_in_mod(e, m, sp, name, ns, dr) {
      none {
        unresolved_err(e, in_mod(m), sp, name, ns_name(ns));
        ret none;
      }
      some(d) { ret some(d); }
    }
}

fn lookup_in_mod(e: env, m: def, sp: span, name: ident, ns: namespace,
                 dr: dir) -> option<def> {
    let defid = def_id_of_def(m);
    if defid.crate != ast::local_crate {
        // examining a module in an external crate
        let cached = e.ext_cache.find({did: defid, ident: name, ns: ns});
        if !is_none(cached) { ret cached; }
        let mut path = [name];
        if defid.node != ast::crate_node_id {
            path = cstore::get_path(e.cstore, defid) + path;
        }
        alt lookup_external(e, defid.crate, path, ns) {
           some(df) {
               e.ext_cache.insert({did: defid, ident: name, ns: ns}, df);
               ret some(df);
           }
           _ { ret none; }
        }
    }
    alt m {
      ast::def_mod(defid) {
        ret lookup_in_local_mod(e, defid.node, sp, name, ns, dr);
      }
      ast::def_native_mod(defid) {
        ret lookup_in_local_native_mod(e, defid.node, sp, name, ns);
      }
      _ {
          // Precondition
          e.sess.span_bug(sp, "lookup_in_mod was passed a non-mod def");
      }
    }
}

fn found_view_item(e: env, id: node_id) -> option<def> {
    alt cstore::find_use_stmt_cnum(e.cstore, id) {
      some(cnum) {
        some(ast::def_mod({crate: cnum, node: ast::crate_node_id}))
      }
      none {
        // This can happen if we didn't load external crate info.
        // Rustdoc depends on this.
        none
      }
    }
}

fn lookup_import(e: env, n_id: node_id, ns: namespace) -> option<def> {
    // Imports are simply ignored when resolving themselves.
    if vec::contains(e.ignored_imports, n_id) { ret none; }
    alt e.imports.get(n_id) {
      todo(name, path, span, scopes) {
        resolve_import(e, n_id, name, *path, span, scopes);
        ret lookup_import(e, n_id, ns);
      }
      resolving(sp) {
        e.sess.span_err(sp, "cyclic import");
        ret none;
      }
      resolved(val, typ, md, _, _, _) {
        if e.used_imports.track {
            e.used_imports.data += [n_id];
        }
        ret alt ns { ns_val { val } ns_type { typ } ns_module { md } };
      }
      is_glob(_,_,_) {
          e.sess.bug("lookup_import: can't handle is_glob");
      }
    }
}

fn lookup_in_local_native_mod(e: env, node_id: node_id, sp: span, id: ident,
                              ns: namespace) -> option<def> {
    ret lookup_in_local_mod(e, node_id, sp, id, ns, inside);
}

fn is_exported(e: env, i: ident, m: @indexed_mod) -> bool {

    alt m.m {
      some(_m) {
        if ast_util::is_exported(i, _m) { ret true; }
      }
      _ {}
    }

    ret vec::contains(m.globbed_exports, i)
        || e.resolve_unexported;
}

// A list search function. Applies `f` to each element of `v`, starting from
// the first. When `f` returns `some(x)`, `list_search` returns `some(x)`. If
// `f` returns `none` for every element, `list_search` returns `none`.
fn list_search<T: copy, U: copy>(ls: @list<T>, f: fn(T) -> option<U>)
        -> option<U> {
    let mut ls = ls;
    loop {
        ls = alt *ls {
          cons(hd, tl) {
            let result = f(hd);
            if !is_none(result) { ret result; }
            tl
          }
          nil { ret none; }
        };
    }
}

fn lookup_in_local_mod(e: env, node_id: node_id, sp: span, id: ident,
                       ns: namespace, dr: dir) -> option<def> {
    let inf = alt e.mod_map.find(node_id) {
            some(x) { x }
            none { e.sess.span_bug(sp, #fmt("lookup_in_local_mod: \
                     module %d not in mod_map", node_id)); }
    };
    if dr == outside && !is_exported(e, id, inf) {
        // if we're in a native mod, then dr==inside, so inf.m is some _mod
        ret none; // name is not visible
    }
    alt inf.index.find(id) {
      none { }
      some(lst) {
        let found = list_search(lst, {|x| lookup_in_mie(e, x, ns)});
        if !is_none(found) {
            ret found;
        }
      }
    }
    // not local or explicitly imported; try globs:
    ret lookup_glob_in_mod(e, inf, sp, id, ns, outside);
}

fn lookup_in_globs(e: env, globs: [glob_imp_def], sp: span, id: ident,
                   ns: namespace, dr: dir) -> option<def> {
    fn lookup_in_mod_(e: env, def: glob_imp_def, sp: span, name: ident,
                      ns: namespace, dr: dir) -> option<glob_imp_def> {
        alt def.path.node {

          ast::view_path_glob(_, id) {
            if vec::contains(e.ignored_imports, id) { ret none; }
          }

          _ {
            e.sess.span_bug(sp, "lookup_in_globs: not a glob");
          }
        }
        alt lookup_in_mod(e, def.def, sp, name, ns, dr) {
          some(d) { option::some({def: d, path: def.path}) }
          none { none }
        }
    }
    let matches = vec::filter_map(copy globs,
                                  {|x| lookup_in_mod_(e, x, sp, id, ns, dr)});
    if vec::len(matches) == 0u {
        ret none;
        }
    else if vec::len(matches) == 1u {
        ret some(matches[0].def);
    } else {
        for matches.each {|match|
            let sp = match.path.span;
            e.sess.span_note(sp, #fmt["'%s' is imported here", id]);
        }
        e.sess.span_fatal(sp, "'" + id + "' is glob-imported from" +
                          " multiple different modules.");
    }
}

fn lookup_glob_in_mod(e: env, info: @indexed_mod, sp: span, id: ident,
                      wanted_ns: namespace, dr: dir) -> option<def> {
    // since we don't know what names we have in advance,
    // absence takes the place of todo()
    if !info.glob_imported_names.contains_key(id) {
        info.glob_imported_names.insert(id, glob_resolving(sp));
        let globs = info.glob_imports;
        let val = lookup_in_globs(e, globs, sp, id, ns_val, dr);
        let typ = lookup_in_globs(e, globs, sp, id, ns_type, dr);
        let md = lookup_in_globs(e, globs, sp, id, ns_module, dr);
        info.glob_imported_names.insert(id, glob_resolved(val, typ, md));
    }
    alt info.glob_imported_names.get(id) {
      glob_resolving(sp) {
          ret none;
      }
      glob_resolved(val, typ, md) {
        ret alt wanted_ns {
          ns_val { val }
          ns_type { typ }
          ns_module { md }
        };
      }
    }
}

fn lookup_in_mie(e: env, mie: mod_index_entry, ns: namespace) ->
   option<def> {
    alt mie {
      mie_view_item(_, id, _) {
         if ns == ns_module { ret found_view_item(e, id); }
      }
      mie_import_ident(id, _) { ret lookup_import(e, id, ns); }
      mie_item(item) { ret found_def_item(item, ns); }
      mie_enum_variant(variant_idx, variants, parent_id, parent_span) {
         alt ns {
            ns_val {
               let vid = variants[variant_idx].node.id;
               ret some(ast::def_variant(local_def(parent_id),
                                        local_def(vid)));
            }
            _ { ret none; }
         }
      }
      mie_native_item(native_item) {
        alt native_item.node {
          ast::native_item_fn(decl, _) {
            if ns == ns_val {
                ret some(ast::def_fn(local_def(native_item.id),
                                     decl.purity));
            }
          }
        }
      }
    }
    ret none;
}


// Module indexing
fn add_to_index(index: hashmap<ident, @list<mod_index_entry>>, id: ident,
                ent: mod_index_entry) {
    alt index.find(id) {
      none { index.insert(id, @cons(ent, @nil)); }
      some(prev) { index.insert(id, @cons(ent, prev)); }
    }
}

fn index_view_items(view_items: [@ast::view_item],
                    index: hashmap<ident, @list<mod_index_entry>>) {
    for view_items.each {|vi|
        alt vi.node {
          ast::view_item_use(ident, _, id) {
           add_to_index(index, ident, mie_view_item(ident, id, vi.span));
          }
          _ {}
        }

        iter_effective_import_paths(*vi) {|vp|
            alt vp.node {
              ast::view_path_simple(ident, _, id) {
                add_to_index(index, ident, mie_import_ident(id, vp.span));
              }
              ast::view_path_list(_, idents, _) {
                for idents.each {|ident|
                    add_to_index(index, ident.node.name,
                                 mie_import_ident(ident.node.id,
                                                  ident.span));
                }
              }

              // globbed imports have to be resolved lazily.
              ast::view_path_glob(_, _) {}
            }
        }
    }
}

fn index_mod(md: ast::_mod) -> mod_index {
    let index = str_hash::<@list<mod_index_entry>>();

    index_view_items(md.view_items, index);

    for md.items.each {|it|
        alt it.node {
          ast::item_const(_, _) | ast::item_fn(_, _, _) | ast::item_mod(_) |
          ast::item_native_mod(_) | ast::item_ty(_, _, _) |
          ast::item_res(*) | ast::item_impl(*) | ast::item_iface(*) {
            add_to_index(index, it.ident, mie_item(it));
          }
          ast::item_enum(variants, _, _) {
            add_to_index(index, it.ident, mie_item(it));
            let mut variant_idx: uint = 0u;
            for variants.each {|v|
                add_to_index(index, v.node.name,
                             mie_enum_variant(variant_idx, variants,
                                             it.id, it.span));
                variant_idx += 1u;
            }
          }
          ast::item_class(tps, _, items, ctor, _, _) {
              // add the class name itself
              add_to_index(index, it.ident, mie_item(it));
              // add the constructor decl
              add_to_index(index, it.ident,
                           mie_item(@{ident: it.ident, attrs: [],
                            id: ctor.node.id,
                            node:
                              item_fn(ctor.node.dec, tps, ctor.node.body),
                            vis: ast::public,
                            span: ctor.node.body.span}));
          }
        }
    }
    ret index;
}


fn index_nmod(md: ast::native_mod) -> mod_index {
    let index = str_hash::<@list<mod_index_entry>>();

    index_view_items(md.view_items, index);

    for md.items.each {|it|
        add_to_index(index, it.ident, mie_native_item(it));
    }
    ret index;
}


// External lookups
fn ns_for_def(d: def) -> namespace {
    alt d {
      ast::def_variant(_, _) { ns_val }
      ast::def_fn(_, _) | ast::def_self(_) |
      ast::def_const(_) | ast::def_arg(_, _) | ast::def_local(_, _) |
      ast::def_upvar(_, _, _) { ns_val }
      ast::def_mod(_) | ast::def_native_mod(_) { ns_module }
      ast::def_ty(_) | ast::def_binding(_) | ast::def_use(_) |
      ast::def_ty_param(_, _) | ast::def_prim_ty(_) | ast::def_class(_)
      { ns_type }
      ast::def_region(_) { fail "regions are not handled by this pass" }
    }
}

fn lookup_external(e: env, cnum: int, ids: [ident], ns: namespace) ->
   option<def> {
    let mut result = none;
    for csearch::lookup_defs(e.sess.cstore, cnum, ids).each {|d|
        e.ext_map.insert(def_id_of_def(d), ids);
        if ns == ns_for_def(d) { result = some(d); }
    }
    ret result;
}


// Collision detection
fn check_for_collisions(e: @env, c: ast::crate) {
    // Module indices make checking those relatively simple -- just check each
    // name for multiple entities in the same namespace.
    for e.mod_map.each_value {|val|
        for val.index.each {|k, v| check_mod_name(*e, k, v); };
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

fn check_mod_name(e: env, name: ident, entries: @list<mod_index_entry>) {
    let mut saw_mod = false;
    let mut saw_type = false;
    let mut saw_value = false;
    let mut entries = entries;
    fn dup(e: env, sp: span, word: str, name: ident) {
        e.sess.span_fatal(sp, "duplicate definition of " + word + name);
    }
    loop {
        alt *entries {
          cons(entry, rest) {
            if !is_none(lookup_in_mie(e, entry, ns_val)) {
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
            entries = rest;
          }
          nil { break; }
        }
    }
}

fn mie_span(mie: mod_index_entry) -> span {
    ret alt mie {
          mie_view_item(_, _, span) { span }
          mie_import_ident(_, span) { span }
          mie_item(item) { item.span }
          mie_enum_variant(_, _, _, span) { span }
          mie_native_item(item) { item.span }
        };
}

fn check_item(e: @env, i: @ast::item, &&x: (), v: vt<()>) {
    fn typaram_names(tps: [ast::ty_param]) -> [ident] {
        let mut x: [ast::ident] = [];
        for tps.each {|tp| x += [tp.ident]; }
        ret x;
    }
    visit::visit_item(i, x, v);
    alt i.node {
      ast::item_fn(decl, ty_params, _) {
        check_fn(*e, i.span, decl);
        ensure_unique(*e, i.span, ty_params, {|tp| tp.ident},
                      "type parameter");
      }
      ast::item_enum(_, ty_params, _) {
        ensure_unique(*e, i.span, ty_params, {|tp| tp.ident},
                      "type parameter");
      }
      ast::item_iface(_, _, methods) {
        ensure_unique(*e, i.span, methods, {|m| m.ident},
                      "method");
      }
      ast::item_impl(_, _, _, _, methods) {
        ensure_unique(*e, i.span, methods, {|m| m.ident},
                      "method");
      }
      _ { }
    }
}

fn check_pat(e: @env, ch: checker, p: @ast::pat) {
    pat_util::pat_bindings(e.def_map, p) {|_i, p_sp, n|
       add_name(ch, p_sp, path_to_ident(n));
    };
}

fn check_arm(e: @env, a: ast::arm, &&x: (), v: vt<()>) {
    visit::visit_arm(a, x, v);
    let ch0 = checker(*e, "binding");
    check_pat(e, ch0, a.pats[0]);
    let seen0 = ch0.seen.get();
    let mut i = vec::len(a.pats);
    while i > 1u {
        i -= 1u;
        let ch = checker(*e, "binding");
        check_pat(e, ch, a.pats[i]);

        // Ensure the bindings introduced in this pattern are the same as in
        // the first pattern.
        if ch.seen.len() != seen0.len() {
            e.sess.span_err(a.pats[i].span,
                            "inconsistent number of bindings");
        } else {
            for ch.seen.each {|name|
                if is_none(vec::find(seen0, bind str::eq(name, _))) {
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
    for b.node.stmts.each {|st|
        alt st.node {
          ast::stmt_decl(d, _) {
            alt d.node {
              ast::decl_local(locs) {
                let local_values = checker(*e, "value");
                for locs.each {|loc|
                     pat_util::pat_bindings(e.def_map, loc.node.pat)
                         {|_i, p_sp, n|
                         let ident = path_to_ident(n);
                         add_name(local_values, p_sp, ident);
                         check_name(values, p_sp, ident);
                     };
                }
              }
              ast::decl_item(it) {
                alt it.node {
                  ast::item_enum(variants, _, _) {
                    add_name(types, it.span, it.ident);
                    for variants.each {|v|
                        add_name(values, v.span, v.node.name);
                    }
                  }
                  ast::item_mod(_) | ast::item_native_mod(_) {
                    add_name(mods, it.span, it.ident);
                  }
                  ast::item_const(_, _) | ast::item_fn(*) {
                    add_name(values, it.span, it.ident);
                  }
                  ast::item_ty(*) | ast::item_iface(*) {
                    add_name(types, it.span, it.ident);
                  }
                  ast::item_res(*) {
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

fn check_fn(e: env, sp: span, decl: ast::fn_decl) {
    fn arg_name(a: ast::arg) -> ident { ret a.ident; }
    ensure_unique(e, sp, decl.inputs, arg_name, "argument");
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

type checker = @{seen: dvec<ident>, kind: str, sess: session};

fn checker(e: env, kind: str) -> checker {
    ret @{seen: dvec(), kind: kind, sess: e.sess};
}

fn check_name(ch: checker, sp: span, name: ident) {
    for ch.seen.each {|s|
        if str::eq(s, name) {
            ch.sess.span_fatal(sp, "duplicate " + ch.kind + " name: " + name);
        }
    }
}
fn add_name(ch: checker, sp: span, name: ident) {
    check_name(ch, sp, name);
    ch.seen.push(name);
}

fn ensure_unique<T>(e: env, sp: span, elts: [T], id: fn(T) -> ident,
                    kind: str) {
    let ch = checker(e, kind);
    for elts.each {|elt| add_name(ch, sp, id(elt)); }
}

fn check_exports(e: @env) {

    fn iter_mod(e: env, m: def, sp: span, _dr: dir,
                f: fn(ident: ident, def: def)) {
        let defid = def_id_of_def(m);

        if defid.crate != ast::local_crate {
            // FIXME: ought to support external export-globs eventually.
            e.sess.span_unimpl(sp, "glob-export of items in external crate");
        } else {

            let mid = def_id_of_def(m);
            assert mid.crate == ast::local_crate;
            let ixm = e.mod_map.get(mid.node);

            for ixm.index.each {|ident, mies|
                list::iter(mies) {|mie|
                    alt mie {
                      mie_item(item) {
                        let defs =
                            [ found_def_item(item, ns_val),
                             found_def_item(item, ns_type),
                             found_def_item(item, ns_module) ];
                        for defs.each {|d|
                            alt d {
                              some(def) {
                                f(ident, def);
                              }
                              _ {}
                            }
                        }
                      }
                      _ {
                        let s = "glob-export from mod with non-items";
                        e.sess.span_unimpl(sp, s);
                      }
                    }
                }
            }
        }
    }



    fn lookup_glob_any(e: @env, info: @indexed_mod, sp: span,
                       ident: ident, export_id: node_id) -> bool {
        let m = lookup_glob_in_mod(*e, info, sp, ident, ns_module, inside);
        let v = lookup_glob_in_mod(*e, info, sp, ident, ns_val, inside);
        let t = lookup_glob_in_mod(*e, info, sp, ident, ns_type, inside);
        maybe_add_reexport(e, export_id, m);
        maybe_add_reexport(e, export_id, v);
        maybe_add_reexport(e, export_id, t);
        is_some(m) || is_some(v) || is_some(t)
    }


    fn maybe_add_reexport(e: @env, export_id: node_id, def: option<def>) {
        option::iter(def) {|def|
            add_export(e, export_id, def_id_of_def(def), true);
        }
    }
    fn add_export(e: @env, export_id: node_id, target_id: def_id,
                  reexp: bool) {
        let found = alt e.exp_map.find(export_id) {
          some(f) { f } none { [] }
        };
        e.exp_map.insert(export_id, found + [{reexp: reexp, id: target_id}]);
    }

    fn check_export(e: @env, ident: str, _mod: @indexed_mod,
                    export_id: node_id, vi: @view_item) {
        let mut found_something = false;
        if _mod.index.contains_key(ident) {
            found_something = true;
            let xs = _mod.index.get(ident);
            list::iter(xs) {|x|
                alt x {
                  mie_import_ident(id, _) {
                    alt check e.imports.get(id) {
                      resolved(v, t, m, _, rid, _) {
                        maybe_add_reexport(e, export_id, v);
                        maybe_add_reexport(e, export_id, t);
                        maybe_add_reexport(e, export_id, m);
                      }
                      _ { e.sess.span_bug(vi.span, "unresolved export"); }
                    }
                  }
                  mie_item(@{id, _}) | mie_native_item(@{id, _}) |
                  mie_enum_variant(_, _, id, _) {
                    add_export(e, export_id, local_def(id), false);
                  }
                  _ { }
                }
            }
        }
        /*
          This code previously used bitwise or (|=) but that was wrong,
          because we need or to be lazy here. If something was already
          found, we don't want to call lookup_glob_any (see #2316 for
          what happens if we do)
         */
        found_something = found_something ||
           lookup_glob_any(e, _mod, vi.span, ident, export_id);
        if !found_something {
            e.sess.span_warn(vi.span,
                             #fmt("exported item %s is not defined", ident));
        }
    }

    fn check_enum_ok(e: @env, sp:span, id: ident, _mod: @indexed_mod)
        -> node_id {
        alt _mod.index.find(id) {
          none {
            e.sess.span_fatal(sp, #fmt("undefined id %s in an export", id));
          }
          some(ms) {
            let maybe_id = list_search(ms) {|m|
                alt m {
                  mie_item(@{node: item_enum(_, _, _), id, _}) { some(id) }
                  _ { none }
                }
            };
            alt maybe_id {
              some(an_id) { an_id }
              _ { e.sess.span_fatal(sp, #fmt("%s does not refer \
                                              to an enumeration", id)); }
            }
          }
        }
    }

    fn check_export_enum_list(e: @env, export_id: node_id, _mod: @indexed_mod,
                              span: codemap::span, id: ast::ident,
                              ids: [ast::path_list_ident]) {
        let parent_id = check_enum_ok(e, span, id, _mod);
        add_export(e, export_id, local_def(parent_id), false);
        for ids.each {|variant_id|
            let mut found = false;
            alt _mod.index.find(variant_id.node.name) {
              some(ms) {
                list::iter(ms) {|m|
                    alt m {
                      mie_enum_variant(_, _, actual_parent_id, _) {
                        found = true;
                        if actual_parent_id != parent_id {
                            e.sess.span_err(
                                span, #fmt("variant %s doesn't belong to \
                                            enum %s",
                                           variant_id.node.name, id));
                        }
                      }
                      _ {}
                    }
                }
              }
              _ {}
            }
            if !found {
                e.sess.span_err(span, #fmt("%s is not a variant",
                                           variant_id.node.name));
            }
        }
    }

    for e.mod_map.each_value {|_mod|
        alt _mod.m {
          some(m) {
            let glob_is_re_exported = int_hash();

            for m.view_items.each {|vi|
                iter_export_paths(*vi) { |vp|
                    alt vp.node {
                      ast::view_path_simple(ident, _, id) {
                        check_export(e, ident, _mod, id, vi);
                      }
                      ast::view_path_list(path, ids, node_id) {
                        let id = if vec::len(path.idents) == 1u {
                            path.idents[0]
                        } else {
                            e.sess.span_fatal(vp.span, "bad export name-list")
                        };
                        check_export_enum_list(e, node_id, _mod, vp.span, id,
                                               ids);
                      }
                      ast::view_path_glob(_, node_id) {
                        glob_is_re_exported.insert(node_id, ());
                      }
                    }
                }
            }
            // Now follow the export-glob links and fill in the
            // globbed_exports and exp_map lists.
            for _mod.glob_imports.each {|glob|
                let id = alt check glob.path.node {
                  ast::view_path_glob(_, node_id) { node_id }
                };
                if ! glob_is_re_exported.contains_key(id) { cont; }
                iter_mod(*e, glob.def,
                         glob.path.span, outside) {|ident, def|
                    _mod.globbed_exports += [ident];
                    maybe_add_reexport(e, id, some(def));
                }
            }
          }
          none { }
        }
    }
}

// Impl resolution

type method_info = {did: def_id, n_tps: uint, ident: ast::ident};
/* An _impl represents an implementation that's currently in scope.
   Its fields:
   * did: the def id of the class or impl item
   * ident: the name of the impl, unless it has no name (as in
   "impl of X") in which case the ident
   is the ident of the iface that's being implemented
   * methods: the item's methods
*/
type _impl = {did: def_id, ident: ast::ident, methods: [@method_info]};
type iscopes = @list<@[@_impl]>;

fn resolve_impls(e: @env, c: @ast::crate) {
    visit::visit_crate(*c, @nil, visit::mk_vt(@{
        visit_block: bind visit_block_with_impl_scope(e, _, _, _),
        visit_mod: bind visit_mod_with_impl_scope(e, _, _, _, _, _),
        visit_expr: bind resolve_impl_in_expr(e, _, _, _)
        with *visit::default_visitor()
    }));
}

fn find_impls_in_view_item(e: env, vi: @ast::view_item,
                           &impls: [@_impl], sc: option<iscopes>) {
    fn lookup_imported_impls(e: env, id: node_id,
                             act: fn(@[@_impl])) {
        alt e.imports.get(id) {
          resolved(_, _, _, is, _, _) { act(is); }
          todo(name, path, span, scopes) {
            resolve_import(e, id, name, *path, span, scopes);
            alt check e.imports.get(id) {
              resolved(_, _, _, is, _, _) { act(is); }
            }
          }
          _ {}
        }
    }

    iter_effective_import_paths(*vi) { |vp|
        alt vp.node {
          ast::view_path_simple(name, pt, id) {
            let mut found = [];
            if vec::len(pt.idents) == 1u {
                option::iter(sc) {|sc|
                    list::iter(sc) {|level|
                        if vec::len(found) == 0u {
                            for vec::each(*level) {|imp|
                                if imp.ident == pt.idents[0] {
                                    found += [@{ident: name with *imp}];
                                }
                            }
                            if vec::len(found) > 0u { impls += found; }
                        }
                    }
                }
            } else {
                lookup_imported_impls(e, id) {|is|
                    for vec::each(*is) {|i|
                        impls += [@{ident: name with *i}];
                    }
                }
            }
          }

          ast::view_path_list(base, names, _) {
            for names.each {|nm|
                lookup_imported_impls(e, nm.node.id) {|is| impls += *is; }
            }
          }

          ast::view_path_glob(ids, id) {
            alt check e.imports.get(id) {
              is_glob(path, sc, sp) {
                alt follow_import(e, sc, *path, sp) {
                  some(def) { find_impls_in_mod(e, def, impls, none); }
                  _ {}
                }
              }
            }
          }
        }
    }
}

/*
  Given an item <i>, adds one record to the mutable vec
  <impls> if the item is an impl; zero or more records if the
  item is a class; and none otherwise. Each record describes
  one interface implemented by i.
 */
fn find_impls_in_item(e: env, i: @ast::item, &impls: [@_impl],
                      name: option<ident>,
                      ck_exports: option<@indexed_mod>) {
    alt i.node {
      ast::item_impl(_, _, ifce, _, mthds) {
        if alt name { some(n) { n == i.ident } _ { true } } &&
           alt ck_exports {
             some(m) { is_exported(e, i.ident, m) }
             _ { true }
           } {
            impls += [@{did: local_def(i.id),
                        ident: i.ident,
                        methods: vec::map(mthds, {|m|
                            @{did: local_def(m.id),
                              n_tps: vec::len(m.tps),
                              ident: m.ident}
                        })}];
        }
      }
      ast::item_class(tps, ifces, items, _, _, _) {
          let (_, mthds) = ast_util::split_class_items(items);
          let n_tps = tps.len();
          vec::iter(ifces) {|p|
              // The def_id, in this case, identifies the combination of
              // class and iface
              impls += [@{did: local_def(p.id),
                         ident: i.ident,
                         methods: vec::map(mthds, {|m|
                                      @{did: local_def(m.id),
                                          n_tps: n_tps + m.tps.len(),
                                          ident: m.ident}})}];
          }
      }
      _ {}
    }
}

fn find_impls_in_mod_by_id(e: env, defid: def_id, &impls: [@_impl],
                           name: option<ident>) {
    let mut cached;
    alt e.impl_cache.find(defid) {
      some(some(v)) { cached = v; }
      some(none) { ret; }
      none {
        e.impl_cache.insert(defid, none);
        cached = if defid.crate == ast::local_crate {
            let mut tmp = [];
            let mi = e.mod_map.get(defid.node);
            let md = option::get(mi.m);
            for md.view_items.each {|vi|
                find_impls_in_view_item(e, vi, tmp, none);
            }
            for md.items.each {|i|
                find_impls_in_item(e, i, tmp, none, none);
            }
            @vec::filter(tmp) {|i| is_exported(e, i.ident, mi)}
        } else {
            csearch::get_impls_for_mod(e.sess.cstore, defid, none)
        };
        e.impl_cache.insert(defid, some(cached));
      }
    }
    alt name {
      some(n) {
        for vec::each(*cached) {|im|
            if n == im.ident { impls += [im]; }
        }
      }
      _ { impls += *cached; }
    }
}

fn find_impls_in_mod(e: env, m: def, &impls: [@_impl],
                     name: option<ident>) {
    alt m {
      ast::def_mod(defid) {
        find_impls_in_mod_by_id(e, defid, impls, name);
      }
      _ {}
    }
}

fn visit_block_with_impl_scope(e: @env, b: ast::blk, &&sc: iscopes,
                               v: vt<iscopes>) {
    let mut impls = [];
    for b.node.view_items.each {|vi|
        find_impls_in_view_item(*e, vi, impls, some(sc));
    }
    for b.node.stmts.each {|st|
        alt st.node {
          ast::stmt_decl(@{node: ast::decl_item(i), _}, _) {
            find_impls_in_item(*e, i, impls, none, none);
          }
          _ {}
        }
    }
    let sc = if vec::len(impls) > 0u { @cons(@impls, sc) } else { sc };
    visit::visit_block(b, sc, v);
}

fn visit_mod_with_impl_scope(e: @env, m: ast::_mod, s: span, id: node_id,
                             &&sc: iscopes, v: vt<iscopes>) {
    let mut impls = [];
    for m.view_items.each {|vi|
        find_impls_in_view_item(*e, vi, impls, some(sc));
    }
    for m.items.each {|i| find_impls_in_item(*e, i, impls, none, none); }
    let impls = @impls;
    visit::visit_mod(m, s, id, if vec::len(*impls) > 0u {
                                   @cons(impls, sc)
                               } else {
                                   sc
                               }, v);
    e.impl_map.insert(id, @cons(impls, @nil));
}

fn resolve_impl_in_expr(e: @env, x: @ast::expr,
                        &&sc: iscopes, v: vt<iscopes>) {
    alt x.node {
      // Store the visible impls in all exprs that might need them
      ast::expr_field(_, _, _) | ast::expr_path(_) | ast::expr_cast(_, _) |
      ast::expr_binary(_, _, _) | ast::expr_unary(_, _) |
      ast::expr_assign_op(_, _, _) | ast::expr_index(_, _) {
        e.impl_map.insert(x.id, sc);
      }
      ast::expr_new(p, _, _) {
        e.impl_map.insert(p.id, sc);
      }
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
