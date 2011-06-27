
import front::ast;
import front::ast::ident;
import front::ast::def;
import front::ast::def_id;
import front::ast::node_id;
import front::ast::local_def;

import metadata::creader;
import metadata::decoder;
import driver::session::session;
import util::common::new_def_hash;
import util::common::new_int_hash;
import util::common::new_str_hash;
import util::common::span;
import util::common::respan;
import middle::ty::constr_table;
import visit::vt;
import std::map::hashmap;
import std::list;
import std::list::list;
import std::list::nil;
import std::list::cons;
import std::option;
import std::option::some;
import std::option::none;
import std::str;
import std::vec;

export resolve_crate;
export def_map;
export crate_map;

// Resolving happens in two passes. The first pass collects defids of all
// (internal) imports and modules, so that they can be looked up when needed,
// and then uses this information to resolve the imports. The second pass
// locates all names (in expressions, types, and alt patterns) and resolves
// them, storing the resulting def in the AST nodes.

// This module internally uses -1 as a def_id for the top_level module in a
// crate. The parser doesn't assign a def_id to this module.
// (FIXME See https://github.com/graydon/rust/issues/358 for the reason this
//  isn't a const.)
tag scope {
    scope_crate(@ast::crate);
    scope_item(@ast::item);
    scope_fn(ast::fn_decl, vec[ast::ty_param]);
    scope_native_item(@ast::native_item);
    scope_loop(@ast::local); // there's only 1 decl per loop.
    scope_block(ast::block);
    scope_arm(ast::arm);
}

type scopes = list[scope];

tag import_state {
    todo(@ast::view_item, scopes); // only used for explicit imports

    resolving(span);
    resolved(option::t[def],

             /* value */
             option::t[def],

             /* type */
             option::t[def]);
    /* module */

}

type ext_hash = hashmap[tup(def_id, str, namespace), def];

fn new_ext_hash() -> ext_hash {
    fn hash(&tup(def_id, str, namespace) v) -> uint {
        ret str::hash(v._1) + util::common::hash_def(v._0) +
                alt (v._2) {
                    case (ns_value) { 1u }
                    case (ns_type) { 2u }
                    case (ns_module) { 3u }
                };
    }
    fn eq(&tup(def_id, str, namespace) v1, &tup(def_id, str, namespace) v2) ->
       bool {
        ret util::common::def_eq(v1._0, v2._0) && str::eq(v1._1, v2._1) &&
                v1._2 == v2._2;
    }
    ret std::map::mk_hashmap[tup(def_id, str, namespace), def](hash, eq);
}

tag mod_index_entry {
    mie_view_item(@ast::view_item);
    mie_item(@ast::item);
    mie_native_item(@ast::native_item);
    mie_tag_variant(@ast::item, /* tag item */uint);
    /* variant index */

}

type mod_index = hashmap[ident, list[mod_index_entry]];

type indexed_mod =
    rec(option::t[ast::_mod] m,
        mod_index index,
        mutable vec[def] glob_imports,
        hashmap[str, import_state] glob_imported_names);


/* native modules can't contain tags, and we don't store their ASTs because we
   only need to look at them to determine exports, which they can't control.*/

// It should be safe to use index to memoize lookups of globbed names.
type crate_map = hashmap[node_id, ast::crate_num];

type def_map = hashmap[node_id, def];

type env =
    rec(crate_map crate_map,
        def_map def_map,
        constr_table fn_constrs,
        ast_map::map ast_map,
        hashmap[ast::node_id, import_state] imports,
        hashmap[ast::node_id, @indexed_mod] mod_map,
        hashmap[def_id, vec[ident]] ext_map,
        ext_hash ext_cache,
        session sess);


// Used to distinguish between lookups from outside and from inside modules,
// since export restrictions should only be applied for the former.
tag dir { inside; outside; }

tag namespace { ns_value; ns_type; ns_module; }

fn resolve_crate(session sess, &ast_map::map amap, @ast::crate crate) ->
   tup(def_map, constr_table) {
    auto e =
        @rec(crate_map=new_int_hash[ast::crate_num](),
             def_map=new_int_hash[def](),
             fn_constrs = new_int_hash[vec[ty::constr_def]](),
             ast_map=amap,
             imports=new_int_hash[import_state](),
             mod_map=new_int_hash[@indexed_mod](),
             ext_map=new_def_hash[vec[ident]](),
             ext_cache=new_ext_hash(),
             sess=sess);
    creader::read_crates(sess, e.crate_map, *crate);
    map_crate(e, crate);
    resolve_imports(*e);
    check_for_collisions(e, *crate);
    resolve_names(e, crate);
    ret tup(e.def_map, e.fn_constrs);
}


// Locate all modules and imports and index them, so that the next passes can
// resolve through them.
fn map_crate(&@env e, &@ast::crate c) {
    // First, find all the modules, and index the names that they contain

    auto v_map_mod =
        @rec(visit_view_item=bind index_vi(e, _, _, _),
             visit_item=bind index_i(e, _, _, _)
             with *visit::default_visitor[scopes]());
    visit::visit_crate(*c, cons(scope_crate(c), @nil),
                       visit::vtor(v_map_mod));
    // Register the top-level mod 

    e.mod_map.insert(-1,
                     @rec(m=some(c.node.module),
                          index=index_mod(c.node.module),
                          mutable glob_imports=vec::empty[def](),
                          glob_imported_names=new_str_hash[import_state]()));
    fn index_vi(@env e, &@ast::view_item i, &scopes sc, &vt[scopes] v) {
        alt (i.node) {
            case (ast::view_item_import(_, ?ids, ?id)) {
                e.imports.insert(id, todo(i, sc));
            }
            case (_) { }
        }
    }
    fn index_i(@env e, &@ast::item i, &scopes sc, &vt[scopes] v) {
        visit_item_with_scope(i, sc, v);
        alt (i.node) {
            case (ast::item_mod(?md)) {
                auto s = new_str_hash[import_state]();
                e.mod_map.insert(i.id,
                                 @rec(m=some(md),
                                      index=index_mod(md),
                                      mutable glob_imports=vec::empty[def](),
                                      glob_imported_names=s));
            }
            case (ast::item_native_mod(?nmd)) {
                auto s = new_str_hash[import_state]();
                e.mod_map.insert(i.id,
                                 @rec(m=none[ast::_mod],
                                      index=index_nmod(nmd),
                                      mutable glob_imports=vec::empty[def](),
                                      glob_imported_names=s));
            }
            case (_) { }
        }
    }
    // Next, assemble the links for globbed imports.

    auto v_link_glob =
        @rec(visit_view_item=bind link_glob(e, _, _, _),
             visit_item=visit_item_with_scope
             with *visit::default_visitor[scopes]());
    visit::visit_crate(*c, cons(scope_crate(c), @nil),
                       visit::vtor(v_link_glob));
    fn link_glob(@env e, &@ast::view_item vi, &scopes sc, &vt[scopes] v) {
        fn find_mod(@env e, scopes sc) -> @indexed_mod {
            alt (sc) {
                case (cons(scope_item(?i), ?tl)) {
                    alt (i.node) {
                        case (ast::item_mod(_)) {
                            ret e.mod_map.get(i.id);
                        }
                        case (ast::item_native_mod(_)) {
                            ret e.mod_map.get(i.id);
                        }
                        case (_) { be find_mod(e, *tl); }
                    }
                }
                case (_) {
                    ret e.mod_map.get(-1); //top-level

                }
            }
        }
        alt (vi.node) {
            case (
                 //if it really is a glob import, that is
                 ast::view_item_import_glob(?path, _)) {
                auto imp = follow_import(*e, sc, path, vi.span);
                if (option::is_some(imp)) {
                    find_mod(e, sc).glob_imports +=
                        [option::get(imp)];
                }
            }
            case (_) { }
        }
    }
}

fn resolve_imports(&env e) {
    for each (@tup(ast::node_id, import_state) it in e.imports.items()) {
        alt (it._1) {
            case (todo(?item, ?sc)) { resolve_import(e, item, sc); }
            case (resolved(_, _, _)) { }
        }
    }
    e.sess.abort_if_errors();
}

fn resolve_names(&@env e, &@ast::crate c) {
    auto v =
        @rec(visit_native_item=visit_native_item_with_scope,
             visit_item=visit_item_with_scope,
             visit_block=visit_block_with_scope,
             visit_arm=bind walk_arm(e, _, _, _),
             visit_expr=bind walk_expr(e, _, _, _),
             visit_ty=bind walk_ty(e, _, _, _),
             visit_constr=bind walk_constr(e, _, _, _),
             visit_fn=bind visit_fn_with_scope(e, _, _, _, _, _, _, _)
             with *visit::default_visitor());
    visit::visit_crate(*c, cons(scope_crate(c), @nil), visit::vtor(v));
    e.sess.abort_if_errors();

    fn walk_expr(@env e, &@ast::expr exp, &scopes sc, &vt[scopes] v) {
        visit_expr_with_scope(exp, sc, v);
        alt (exp.node) {
            case (ast::expr_path(?p)) {
                maybe_insert(e, exp.id,
                             lookup_path_strict(*e, sc, exp.span,
                                                p.node.idents, ns_value));
            }
            case (_) { }
        }
    }
    fn walk_ty(@env e, &@ast::ty t, &scopes sc, &vt[scopes] v) {
        visit::visit_ty(t, sc, v);
        alt (t.node) {
            case (ast::ty_path(?p, ?id)) {
                maybe_insert(e, id,
                             lookup_path_strict(*e, sc, t.span,
                                                p.node.idents, ns_type));
            }
            case (_) { }
        }
    }
    fn walk_constr(@env e, &@ast::constr c, &scopes sc, &vt[scopes] v) {
        maybe_insert(e, c.node.id,
                     lookup_path_strict(*e, sc, c.span,
                                        c.node.path.node.idents, ns_value));
    }
    fn walk_arm(@env e, &ast::arm a, &scopes sc, &vt[scopes] v) {
        walk_pat(*e, sc, a.pat);
        visit_arm_with_scope(a, sc, v);
    }
    fn walk_pat(&env e, &scopes sc, &@ast::pat pat) {
        alt (pat.node) {
            case (ast::pat_tag(?p, ?children, ?id)) {
                auto fnd =
                    lookup_path_strict(e, sc, p.span, p.node.idents,
                                       ns_value);
                if (option::is_some(fnd)) {
                    alt (option::get(fnd)) {
                        case (ast::def_variant(?did, ?vid)) {
                            e.def_map.insert(id, option::get(fnd));
                            for (@ast::pat child in children) {
                                walk_pat(e, sc, child);
                            }
                        }
                        case (_) {
                            e.sess.span_err(p.span,
                                            "not a tag variant: " +
                                            ast::path_name(p));
                        }
                    }
                }
            }
            case (_) { }
        }
    }

    fn maybe_insert(@env e, node_id id,
                    option::t[def] def) {
        if (option::is_some(def)) {
            e.def_map.insert(id, option::get(def));
        }
    }
}


// Visit helper functions
fn visit_item_with_scope(&@ast::item i, &scopes sc, &vt[scopes] v) {
    visit::visit_item(i, cons(scope_item(i), @sc), v);
}

fn visit_native_item_with_scope(&@ast::native_item ni, &scopes sc,
                                &vt[scopes] v) {
    visit::visit_native_item(ni, cons(scope_native_item(ni), @sc), v);
}

fn visit_fn_with_scope(&@env e, &ast::_fn f, &vec[ast::ty_param] tp, &span sp,
                       &ident name, node_id id, &scopes sc,
                       &vt[scopes] v) {
    // here's where we need to set up the mapping
    // for f's constrs in the table.

    for (@ast::constr c in f.decl.constraints) {
        resolve_constr(e, id, c, sc, v); 
    }
    visit::visit_fn(f, tp, sp, name, id,
                    cons(scope_fn(f.decl, tp), @sc), v);
}

fn visit_block_with_scope(&ast::block b, &scopes sc, &vt[scopes] v) {
    visit::visit_block(b, cons(scope_block(b), @sc), v);
}

fn visit_arm_with_scope(&ast::arm a, &scopes sc, &vt[scopes] v) {
    visit::visit_arm(a, cons(scope_arm(a), @sc), v);
}

fn visit_expr_with_scope(&@ast::expr x, &scopes sc, &vt[scopes] v) {
    auto new_sc =
        alt (x.node) {
            case (ast::expr_for(?d, _, _)) {
                cons[scope](scope_loop(d), @sc)
            }
            case (ast::expr_for_each(?d, _, _)) {
                cons[scope](scope_loop(d), @sc)
            }
            case (ast::expr_fn(?f)) { cons(scope_fn(f.decl, []), @sc) }
            case (_) { sc }
        };
    visit::visit_expr(x, new_sc, v);
}

fn follow_import(&env e, &scopes sc,
                 vec[ident] path, &span sp) -> option::t[def] {
    auto path_len = vec::len(path);
    auto dcur = lookup_in_scope_strict(e, sc, sp, path.(0), ns_module);
    auto i = 1u;
    while (true && option::is_some(dcur)) {
        if (i == path_len) { break; }
        dcur =
            lookup_in_mod_strict(e, option::get(dcur),
                                 sp, path.(i), ns_module, outside);
        i += 1u;
    }
    if (i == path_len) {
        alt (option::get(dcur)) {
            case (ast::def_mod(?def_id)) { ret dcur; }
            case (ast::def_native_mod(?def_id)) { ret dcur; }
            case (_) {
                e.sess.span_err(sp,
                                str::connect(path, "::") +
                                " does not name a module.");
                ret none;
            }
        }
    } else {
        ret none;
    }
}

fn resolve_constr(@env e, node_id id, &@ast::constr c, &scopes sc,
                  &vt[scopes] v) {
    auto new_def =
        lookup_path_strict(*e, sc, c.span, c.node.path.node.idents, ns_value);
    if (option::is_some(new_def)) {
        alt (option::get(new_def)) {
            case (ast::def_fn(?pred_id, ast::pure_fn)) {
                let ty::constr_general[uint] c_ =
                    rec(path=c.node.path, args=c.node.args, id=pred_id);
                let ty::constr_def new_constr = respan(c.span, c_);
                add_constr(e, id, new_constr);
            }
            case (_) {
                e.sess.span_err(c.span,
                                "Non-predicate in constraint: " +
                                ty::path_to_str(c.node.path));
            }
        }
    }
}

fn add_constr(&@env e, node_id id, &ty::constr_def c) {
    e.fn_constrs.insert(id,
                        alt (e.fn_constrs.find(id)) {
                            case (none) { [c] }
                            case (some(?cs)) { cs + [c] }
                        });
}


// Import resolution
fn resolve_import(&env e, &@ast::view_item it, &scopes sc) {
    auto defid;
    auto ids;
    alt (it.node) {
        case (ast::view_item_import(_, ?_ids, ?_id)) {
            defid = local_def(_id);
            ids = _ids;
        }
    }
    e.imports.insert(defid._1, resolving(it.span));
    auto n_idents = vec::len(ids);
    auto end_id = ids.(n_idents - 1u);
    if (n_idents == 1u) {
        auto next_sc = std::list::cdr(sc);
        register(e, defid, it.span, end_id,
                 lookup_in_scope(e, next_sc, it.span, end_id, ns_value),
                 lookup_in_scope(e, next_sc, it.span, end_id, ns_type),
                 lookup_in_scope(e, next_sc, it.span, end_id, ns_module));
        remove_if_unresolved(e.imports, defid._1);
    } else {
        auto dcur = alt(lookup_in_scope(e, sc, it.span, ids.(0), ns_module)) {
            case (some(?dcur)) {
                dcur
            }
            case (none) {
                unresolved_err(e, it.span, ids.(0), ns_name(ns_module));
                remove_if_unresolved(e.imports, defid._1);
                ret () // FIXME (issue #521)
            }
        };
        auto i = 1u;
        while (true) {
            if (i == n_idents - 1u) {
                register(e, defid, it.span, end_id,
                         lookup_in_mod(e, dcur, it.span, end_id, ns_value,
                                       outside),
                         lookup_in_mod(e, dcur, it.span, end_id, ns_type,
                                       outside),
                         lookup_in_mod(e, dcur, it.span, end_id, ns_module,
                                       outside));
                remove_if_unresolved(e.imports, defid._1);
                break;
            } else {
                dcur = alt (lookup_in_mod(e, dcur, it.span, ids.(i),
                                          ns_module, outside)) {
                    case (some(?dcur)) {
                        dcur
                    }
                    case (none) {
                        unresolved_err(e, it.span, ids.(i),
                                       ns_name(ns_module));
                        remove_if_unresolved(e.imports, defid._1);
                        ret () // FIXME (issue #521)
                    }
                };
                i += 1u;
            }
        }
    }
    fn register(&env e, def_id defid, &span sp, &ident name,
                &option::t[def] val, &option::t[def] typ,
                &option::t[def] md) {
        if (option::is_none(val) && option::is_none(typ) &&
                option::is_none(md)) {
            unresolved_err(e, sp, name, "import");
        } else {
            e.imports.insert(defid._1, resolved(val, typ, md));
        }
    }
    fn remove_if_unresolved(hashmap[ast::node_id, import_state] imports,
                            ast::node_id node_id) {
        // If we couldn't resolve the import, don't leave it in a partially
        // resolved state, to avoid having it reported later as a cyclic
        // import
        if (imports.contains_key(node_id)) {
            alt (imports.get(node_id)) {
                case (resolving(_)) {
                    imports.remove(node_id);
                }
                case (_) { }
            }
        }
    }
}


// Utilities
fn ns_name(namespace ns) -> str {
    alt (ns) {
        case (ns_type) { ret "typename"; }
        case (ns_value) { ret "name"; }
        case (ns_module) { ret "modulename"; }
    }
}

fn unresolved_err(&env e, &span sp, &ident name, &str kind) {
    e.sess.span_err(sp, mk_unresolved_msg(name, kind));
}

fn unresolved_fatal(&env e, &span sp, &ident id, &str kind) -> ! {
    e.sess.span_fatal(sp, mk_unresolved_msg(id, kind));
}

fn mk_unresolved_msg(&ident id, &str kind) -> str {
    ret #fmt("unresolved %s: %s", kind, id);
}

// Lookup helpers
fn lookup_path_strict(&env e, &scopes sc, &span sp, vec[ident] idents,
                      namespace ns) -> option::t[def] {
    auto n_idents = vec::len(idents);
    auto headns = if (n_idents == 1u) { ns } else { ns_module };
    auto dcur = lookup_in_scope_strict(e, sc, sp, idents.(0), headns);
    auto i = 1u;
    while (i < n_idents && option::is_some(dcur)) {
        auto curns = if (n_idents == i + 1u) { ns } else { ns_module };
        dcur = lookup_in_mod_strict(e, option::get(dcur),
                                    sp, idents.(i), curns, outside);
        i += 1u;
    }
    ret dcur;
}

fn lookup_in_scope_strict(&env e, scopes sc, &span sp, &ident name,
                          namespace ns) -> option::t[def] {
    alt (lookup_in_scope(e, sc, sp, name, ns)) {
        case (none) {
            unresolved_err(e, sp, name, ns_name(ns));
            ret none;
        }
        case (some(?d)) { ret some(d); }
    }
}

fn scope_is_fn(&scope sc) -> bool {
    ret alt (sc) {
            case (scope_fn(_, _)) { true }
            case (scope_native_item(_)) { true }
            case (_) { false }
        };
}

fn def_is_local(&def d) -> bool {
    ret alt (d) {
            case (ast::def_arg(_)) { true }
            case (ast::def_local(_)) { true }
            case (ast::def_binding(_)) { true }
            case (_) { false }
        };
}

fn def_is_obj_field(&def d) -> bool {
    ret alt (d) { case (ast::def_obj_field(_)) { true } case (_) { false } };
}

fn lookup_in_scope(&env e, scopes sc, &span sp, &ident name, namespace ns) ->
   option::t[def] {
    fn in_scope(&env e, &span sp, &ident name, &scope s, namespace ns) ->
       option::t[def] {
        //not recursing through globs

        alt (s) {
            case (scope_crate(?c)) {
                ret lookup_in_local_mod(e, -1, sp, name, ns, inside);
            }
            case (scope_item(?it)) {
                alt (it.node) {
                    case (ast::item_obj(?ob, ?ty_params, _)) {
                        ret lookup_in_obj(name, ob, ty_params, ns);
                    }
                    case (ast::item_tag(_, ?ty_params)) {
                        if (ns == ns_type) {
                            ret lookup_in_ty_params(name, ty_params);
                        }
                    }
                    case (ast::item_mod(_)) {
                        ret lookup_in_local_mod
                            (e, it.id, sp, name, ns, inside);
                    }
                    case (ast::item_native_mod(?m)) {
                        ret lookup_in_local_native_mod
                            (e, it.id, sp, name, ns);
                    }
                    case (ast::item_ty(_, ?ty_params)) {
                        if (ns == ns_type) {
                            ret lookup_in_ty_params(name, ty_params);
                        }
                    }
                    case (_) { }
                }
            }
            case (scope_native_item(?it)) {
                alt (it.node) {
                    case (ast::native_item_fn(_, ?decl, ?ty_params))
                         {
                        ret lookup_in_fn(name, decl, ty_params, ns);
                    }
                }
            }
            case (scope_fn(?decl, ?ty_params)) {
                ret lookup_in_fn(name, decl, ty_params, ns);
            }
            case (scope_loop(?local)) {
                if (ns == ns_value) {
                    if (str::eq(local.node.ident, name)) {
                        ret some(ast::def_local(local_def(local.node.id)));
                    }
                }
            }
            case (scope_block(?b)) { ret lookup_in_block(name, b.node, ns); }
            case (scope_arm(?a)) {
                if (ns == ns_value) { ret lookup_in_pat(name, *a.pat); }
            }
        }
        ret none[def];
    }
    auto left_fn = false;
    // Used to determine whether obj fields are in scope

    auto left_fn_level2 = false;
    while (true) {
        alt ({ sc }) {
            case (nil) { ret none[def]; }
            case (cons(?hd, ?tl)) {
                auto fnd = in_scope(e, sp, name, hd, ns);
                if (!option::is_none(fnd)) {
                    auto df = option::get(fnd);
                    if (left_fn && def_is_local(df) ||
                            left_fn_level2 && def_is_obj_field(df)) {
                        e.sess.span_fatal(sp,
                                        "attempted dynamic \
                                         environment-capture");
                    }
                    ret fnd;
                }
                if (left_fn) { left_fn_level2 = true; }
                if (ns == ns_value && !left_fn) { left_fn = scope_is_fn(hd); }
                sc = *tl;
            }
        }
    }
    e.sess.bug("reached unreachable code in lookup_in_scope"); // sigh

}

fn lookup_in_ty_params(&ident name, &vec[ast::ty_param] ty_params) ->
   option::t[def] {
    auto i = 0u;
    for (ast::ty_param tp in ty_params) {
        if (str::eq(tp, name)) { ret some(ast::def_ty_arg(i)); }
        i += 1u;
    }
    ret none[def];
}

fn lookup_in_pat(&ident name, &ast::pat pat) -> option::t[def] {
    alt (pat.node) {
        case (ast::pat_bind(?p_name, ?id)) {
            if (str::eq(p_name, name)) {
                ret some(ast::def_binding(local_def(id)));
            }
        }
        case (ast::pat_wild(_)) { }
        case (ast::pat_lit(_, _)) { }
        case (ast::pat_tag(_, ?pats, _)) {
            for (@ast::pat p in pats) {
                auto found = lookup_in_pat(name, *p);
                if (!option::is_none(found)) { ret found; }
            }
        }
    }
    ret none[def];
}

fn lookup_in_fn(&ident name, &ast::fn_decl decl,
                &vec[ast::ty_param] ty_params,
                namespace ns) -> option::t[def] {
    alt (ns) {
        case (ns_value) {
            for (ast::arg a in decl.inputs) {
                if (str::eq(a.ident, name)) {
                    ret some(ast::def_arg(local_def(a.id)));
                }
            }
            ret none[def];
        }
        case (ns_type) { ret lookup_in_ty_params(name, ty_params); }
        case (_) { ret none[def]; }
    }
}

fn lookup_in_obj(&ident name, &ast::_obj ob, &vec[ast::ty_param] ty_params,
                 namespace ns) -> option::t[def] {
    alt (ns) {
        case (ns_value) {
            for (ast::obj_field f in ob.fields) {
                if (str::eq(f.ident, name)) {
                    ret some(ast::def_obj_field(local_def(f.id)));
                }
            }
            ret none[def];
        }
        case (ns_type) { ret lookup_in_ty_params(name, ty_params); }
        case (_) { ret none[def]; }
    }
}

fn lookup_in_block(&ident name, &ast::block_ b, namespace ns) ->
   option::t[def] {
    for (@ast::stmt st in b.stmts) {
        alt (st.node) {
            case (ast::stmt_decl(?d, _)) {
                alt (d.node) {
                    case (ast::decl_local(?loc)) {
                        if (ns == ns_value && str::eq(name, loc.node.ident)) {
                            ret some(ast::def_local(local_def(loc.node.id)));
                        }
                    }
                    case (ast::decl_item(?it)) {
                        alt (it.node) {
                            case (ast::item_tag(?variants, _)) {
                                if (ns == ns_type) {
                                    if (str::eq(it.ident, name)) {
                                        ret some(ast::def_ty
                                                 (local_def(it.id)));
                                    }
                                } else if (ns == ns_value) {
                                    for (ast::variant v in variants) {
                                        if (str::eq(v.node.name, name)) {
                                            auto i = v.node.id;
                                            ret some(ast::def_variant
                                                     (local_def(it.id),
                                                      local_def(i)));
                                        }
                                    }
                                }
                            }
                            case (_) {
                                if (str::eq(it.ident, name)) {
                                    auto found = found_def_item(it, ns);
                                    if (!option::is_none(found)) {
                                        ret found;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            case (_) { }
        }
    }
    ret none[def];
}

fn found_def_item(&@ast::item i, namespace ns) -> option::t[def] {
    alt (i.node) {
        case (ast::item_const(_, _)) {
            if (ns == ns_value) {
                ret some(ast::def_const(local_def(i.id)));
            }
        }
        case (ast::item_fn(?f, _)) {
            if (ns == ns_value) {
                ret some(ast::def_fn(local_def(i.id), f.decl.purity));
            }
        }
        case (ast::item_mod(_)) {
            if (ns == ns_module) {
                ret some(ast::def_mod(local_def(i.id)));
            }
        }
        case (ast::item_native_mod(_)) {
            if (ns == ns_module) {
                ret some(ast::def_native_mod(local_def(i.id)));
            }
        }
        case (ast::item_ty(_, _)) {
            if (ns == ns_type) {
                ret some(ast::def_ty(local_def(i.id)));
            }
        }
        case (ast::item_res(_, _, _, ?ctor_id)) {
            alt (ns) {
                case (ns_value) { ret some(ast::def_fn(local_def(ctor_id),
                                                       ast::impure_fn)); }
                case (ns_type) { ret some(ast::def_ty(local_def(i.id))); }
                case (_) { }
            }
        }
        case (ast::item_tag(_, _)) {
            if (ns == ns_type) {
                ret some(ast::def_ty(local_def(i.id)));
            }
        }
        case (ast::item_obj(_, _, ?ctor_id)) {
            alt (ns) {
                case (ns_value) { ret some(ast::def_fn(local_def(ctor_id),
                                                       ast::impure_fn)); }
                case (ns_type) { ret some(ast::def_ty(local_def(i.id))); }
                case (_) { }
            }
        }
        case (_) { }
    }
    ret none[def];
}

fn lookup_in_mod_strict(&env e, def m, &span sp, &ident name, namespace ns,
                        dir dr) -> option::t[def] {
    alt (lookup_in_mod(e, m, sp, name, ns, dr)) {
        case (none) {
            unresolved_err(e, sp, name, ns_name(ns));
            ret none;
        }
        case (some(?d)) { ret some(d); }
    }
}

fn lookup_in_mod(&env e, def m, &span sp, &ident name, namespace ns,
                 dir dr) ->
   option::t[def] {
    auto defid = ast::def_id_of_def(m);
    if (defid._0 != ast::local_crate) {
        // examining a module in an external crate

        auto cached = e.ext_cache.find(tup(defid, name, ns));
        if (!option::is_none(cached)) { ret cached; }
        auto path = [name];
        if (defid._1 != -1) { path = e.ext_map.get(defid) + path; }
        auto fnd = lookup_external(e, defid._0, path, ns);
        if (!option::is_none(fnd)) {
            e.ext_cache.insert(tup(defid, name, ns), option::get(fnd));
        }
        ret fnd;
    }
    alt (m) {
        case (ast::def_mod(?defid)) {
            ret lookup_in_local_mod(e, defid._1, sp, name, ns, dr);
        }
        case (ast::def_native_mod(?defid)) {
            ret lookup_in_local_native_mod(e, defid._1, sp, name, ns);
        }
    }
}

fn found_view_item(&env e, @ast::view_item vi, namespace ns) ->
   option::t[def] {
    alt (vi.node) {
        case (ast::view_item_use(_, _, ?id)) {
            ret some(ast::def_mod(tup(e.crate_map.get(id), -1)));
        }
        case (ast::view_item_import(_, _, ?id)) {
            ret lookup_import(e, local_def(id), ns);
        }
        case (ast::view_item_import_glob(_, ?defid)) {
            ret none[def]; //will be handled in the fallback glob pass

        }
    }
}

fn lookup_import(&env e, def_id defid, namespace ns) -> option::t[def] {
    alt (e.imports.get(defid._1)) {
        case (todo(?item, ?sc)) {
            resolve_import(e, item, sc);
            ret lookup_import(e, defid, ns);
        }
        case (resolving(?sp)) {
            e.sess.span_err(sp, "cyclic import");
            ret none;
        }
        case (resolved(?val, ?typ, ?md)) {
            ret alt (ns) {
                    case (ns_value) { val }
                    case (ns_type) { typ }
                    case (ns_module) { md }
                };
        }
    }
}

fn lookup_in_local_native_mod(&env e, node_id node_id, &span sp, &ident id,
                              namespace ns) -> option::t[def] {
    ret lookup_in_local_mod(e, node_id, sp, id, ns, inside);
}

fn lookup_in_local_mod(&env e, node_id node_id, &span sp, &ident id,
                       namespace ns, dir dr) -> option::t[def] {
    auto info = e.mod_map.get(node_id);
    if (dr == outside && !ast::is_exported(id, option::get(info.m))) {
        // if we're in a native mod, then dr==inside, so info.m is some _mod

        ret none[def]; // name is not visible

    }
    alt (info.index.find(id)) {
        case (none) { }
        case (some(?lst_)) {
            auto lst = lst_;
            while (true) {
                alt (lst) {
                    case (nil) { break; }
                    case (cons(?hd, ?tl)) {
                        auto found = lookup_in_mie(e, hd, ns);
                        if (!option::is_none(found)) { ret found; }
                        lst = *tl;
                    }
                }
            }
        }
    }

    // not local or explicitly imported; try globs:
    ret lookup_glob_in_mod(e, info, sp, id, ns, outside);
}

fn lookup_glob_in_mod(&env e, @indexed_mod info, &span sp, &ident id,
                      namespace wanted_ns, dir dr) -> option::t[def] {
    fn per_ns(&env e, @indexed_mod info, &span sp, &ident id, namespace ns,
              dir dr) -> option::t[def] {
        fn l_i_m_r(&env e, &def m, &span sp, &ident id, namespace ns, dir dr)
           -> option::t[def] {
            be lookup_in_mod(e, m, sp, id, ns, dr);
        }
        auto matches =
            vec::filter_map[def,
                            def](bind l_i_m_r(e, _, sp, id, ns, dr),
                                 { info.glob_imports });
        if (vec::len(matches) == 0u) {
            ret none[def];
        } else if (vec::len(matches) == 1u) {
            ret some[def](matches.(0));
        } else {
            for (def match in matches) {
                auto span = alt (e.ast_map.get(ast::def_id_of_def(match)._1)){
                    case (ast_map::node_item(?it)) { it.span }
                    case (ast_map::node_obj_ctor(?it)) { it.span }
                    case (ast_map::node_native_item(?it)) { it.span }
                };
                e.sess.span_note(span, "'" + id + "' is defined here.");
            }
            e.sess.span_fatal(sp,
                            "'" + id + "' is glob-imported from" +
                                " multiple different modules.");
        }
    }
    // since we don't know what names we have in advance,
    // absence takes the place of todo()

    if (!info.glob_imported_names.contains_key(id)) {
        info.glob_imported_names.insert(id, resolving(sp));
        auto val = per_ns(e, info, sp, id, ns_value, dr);
        auto typ = per_ns(e, info, sp, id, ns_type, dr);
        auto md = per_ns(e, info, sp, id, ns_module, dr);
        info.glob_imported_names.insert(id, resolved(val, typ, md));
    }
    alt (info.glob_imported_names.get(id)) {
        case (todo(_, _)) { e.sess.bug("Shouldn't've put a todo in."); }
        case (resolving(?sp)) {
            ret none[def]; //circularity is okay in import globs

        }
        case (resolved(?val, ?typ, ?md)) {
            ret alt (wanted_ns) {
                    case (ns_value) { val }
                    case (ns_type) { typ }
                    case (ns_module) { md }
                };
        }
    }
}

fn lookup_in_mie(&env e, &mod_index_entry mie, namespace ns) ->
   option::t[def] {
    alt (mie) {
        case (mie_view_item(?view_item)) {
            ret found_view_item(e, view_item, ns);
        }
        case (mie_item(?item)) { ret found_def_item(item, ns); }
        case (mie_tag_variant(?item, ?variant_idx)) {
            alt (item.node) {
                case (ast::item_tag(?variants, _)) {
                    if (ns == ns_value) {
                        auto vid = variants.(variant_idx).node.id;
                        ret some(ast::def_variant(local_def(item.id),
                                                  local_def(vid)));
                    } else { ret none[def]; }
                }
            }
        }
        case (mie_native_item(?native_item)) {
            alt (native_item.node) {
                case (ast::native_item_ty) {
                    if (ns == ns_type) {
                        ret some(ast::def_native_ty
                                 (local_def(native_item.id)));
                    }
                }
                case (ast::native_item_fn(_, _, _)) {
                    if (ns == ns_value) {
                        ret some(ast::def_native_fn
                                 (local_def(native_item.id)));
                    }
                }
            }
        }
        case (_) { }
    }
    ret none[def];
}


// Module indexing
fn add_to_index(&hashmap[ident, list[mod_index_entry]] index, &ident id,
                &mod_index_entry ent) {
    alt (index.find(id)) {
        case (none) { index.insert(id, cons(ent, @nil[mod_index_entry])); }
        case (some(?prev)) { index.insert(id, cons(ent, @prev)); }
    }
}

fn index_mod(&ast::_mod md) -> mod_index {
    auto index = new_str_hash[list[mod_index_entry]]();
    for (@ast::view_item it in md.view_items) {
        alt (it.node) {
            case (ast::view_item_use(?ident, _, _)) {
                add_to_index(index, ident, mie_view_item(it));
            }
            case (ast::view_item_import(?def_ident, _, _)) {
                add_to_index(index, def_ident, mie_view_item(it));
            }
            case (
                 //globbed imports have to be resolved lazily.
                 ast::view_item_import_glob(_, _)) {
            }
            case (ast::view_item_export(_, _)) { }
        }
    }
    for (@ast::item it in md.items) {
        alt (it.node) {
            case (ast::item_const(_, _)) {
                add_to_index(index, it.ident, mie_item(it));
            }
            case (ast::item_fn(_, _)) {
                add_to_index(index, it.ident, mie_item(it));
            }
            case (ast::item_mod(_)) {
                add_to_index(index, it.ident, mie_item(it));
            }
            case (ast::item_native_mod(_)) {
                add_to_index(index, it.ident, mie_item(it));
            }
            case (ast::item_ty(_, _)) {
                add_to_index(index, it.ident, mie_item(it));
            }
            case (ast::item_res(_, _, _, _)) {
                add_to_index(index, it.ident, mie_item(it));
            }
            case (ast::item_tag(?variants, _)) {
                add_to_index(index, it.ident, mie_item(it));
                let uint variant_idx = 0u;
                for (ast::variant v in variants) {
                    add_to_index(index, v.node.name,
                                 mie_tag_variant(it, variant_idx));
                    variant_idx += 1u;
                }
            }
            case (ast::item_obj(_, _, _)) {
                add_to_index(index, it.ident, mie_item(it));
            }
        }
    }
    ret index;
}

fn index_nmod(&ast::native_mod md) -> mod_index {
    auto index = new_str_hash[list[mod_index_entry]]();
    for (@ast::view_item it in md.view_items) {
        alt (it.node) {
            case (ast::view_item_import(?def_ident, _, _)) {
                add_to_index(index, def_ident, mie_view_item(it));
            }
            case (ast::view_item_import_glob(_, _)) { }
            case (ast::view_item_export(_, _)) { }
        }
    }
    for (@ast::native_item it in md.items) {
        add_to_index(index, it.ident, mie_native_item(it));
    }
    ret index;
}


// External lookups
fn ns_for_def(def d) -> namespace {
    ret alt (d) {
            case (ast::def_fn(?id, _)) { ns_value }
            case (ast::def_obj_field(?id)) { ns_value }
            case (ast::def_mod(?id)) { ns_module }
            case (ast::def_native_mod(?id)) { ns_module }
            case (ast::def_const(?id)) { ns_value }
            case (ast::def_arg(?id)) { ns_value }
            case (ast::def_local(?id)) { ns_value }
            case (ast::def_variant(_, ?id)) { ns_value }
            case (ast::def_ty(?id)) { ns_type }
            case (ast::def_binding(?id)) { ns_type }
            case (ast::def_use(?id)) { ns_module }
            case (ast::def_native_ty(?id)) { ns_type }
            case (ast::def_native_fn(?id)) { ns_value }
        };
}

fn lookup_external(&env e, int cnum, vec[ident] ids, namespace ns) ->
   option::t[def] {
    for (def d in decoder::lookup_defs(e.sess, cnum, ids)) {
        e.ext_map.insert(ast::def_id_of_def(d), ids);
        if (ns == ns_for_def(d)) { ret some(d); }
    }
    ret none[def];
}


// Collision detection
fn check_for_collisions(&@env e, &ast::crate c) {
    // Module indices make checking those relatively simple -- just check each
    // name for multiple entities in the same namespace.

    for each (@tup(ast::node_id, @indexed_mod) m in e.mod_map.items()) {
        for each (@tup(ident, list[mod_index_entry]) name in
                 m._1.index.items()) {
            check_mod_name(*e, name._0, name._1);
        }
    }
    // Other scopes have to be checked the hard way.

    auto v =
        @rec(visit_item=bind check_item(e, _, _, _),
             visit_block=bind check_block(e, _, _, _),
             visit_arm=bind check_arm(e, _, _, _)
             with *visit::default_visitor());
    visit::visit_crate(c, (), visit::vtor(v));
}

fn check_mod_name(&env e, &ident name, list[mod_index_entry] entries) {
    auto saw_mod = false;
    auto saw_type = false;
    auto saw_value = false;
    fn dup(&env e, &span sp, &str word, &ident name) {
        e.sess.span_fatal(sp, "duplicate definition of " + word + name);
    }
    while (true) {
        alt (entries) {
            case (cons(?entry, ?rest)) {
                if (!option::is_none(lookup_in_mie(e, entry, ns_value))) {
                    if (saw_value) {
                        dup(e, mie_span(entry), "", name);
                    } else { saw_value = true; }
                }
                if (!option::is_none(lookup_in_mie(e, entry, ns_type))) {
                    if (saw_type) {
                        dup(e, mie_span(entry), "type ", name);
                    } else { saw_type = true; }
                }
                if (!option::is_none(lookup_in_mie(e, entry, ns_module))) {
                    if (saw_mod) {
                        dup(e, mie_span(entry), "module ", name);
                    } else { saw_mod = true; }
                }
                entries = *rest;
            }
            case (nil) { break; }
        }
    }
}

fn mie_span(&mod_index_entry mie) -> span {
    alt (mie) {
        case (mie_view_item(?item)) { ret item.span; }
        case (mie_item(?item)) { ret item.span; }
        case (mie_tag_variant(?item, _)) { ret item.span; }
        case (mie_native_item(?item)) { ret item.span; }
    }
}

fn check_item(@env e, &@ast::item i, &() x, &vt[()] v) {
    visit::visit_item(i, x, v);
    alt (i.node) {
        case (ast::item_fn(?f, ?ty_params)) {
            check_fn(*e, i.span, f);
            ensure_unique(*e, i.span, ty_params, ident_id, "type parameter");
        }
        case (ast::item_obj(?ob, ?ty_params, _)) {
            fn field_name(&ast::obj_field field) -> ident { ret field.ident; }
            ensure_unique(*e, i.span, ob.fields, field_name, "object field");
            for (@ast::method m in ob.methods) {
                check_fn(*e, m.span, m.node.meth);
            }
            ensure_unique(*e, i.span, ty_params, ident_id, "type parameter");
        }
        case (ast::item_tag(_, ?ty_params)) {
            ensure_unique(*e, i.span, ty_params, ident_id, "type parameter");
        }
        case (_) { }
    }
}

fn check_arm(@env e, &ast::arm a, &() x, &vt[()] v) {
    visit::visit_arm(a, x, v);
    fn walk_pat(checker ch, &@ast::pat p) {
        alt (p.node) {
            case (ast::pat_bind(?name, _)) { add_name(ch, p.span, name); }
            case (ast::pat_tag(_, ?children, _)) {
                for (@ast::pat child in children) { walk_pat(ch, child); }
            }
            case (_) { }
        }
    }
    walk_pat(checker(*e, "binding"), a.pat);
}

fn check_block(@env e, &ast::block b, &() x, &vt[()] v) {
    visit::visit_block(b, x, v);
    auto values = checker(*e, "value");
    auto types = checker(*e, "type");
    auto mods = checker(*e, "module");
    for (@ast::stmt st in b.node.stmts) {
        alt (st.node) {
            case (ast::stmt_decl(?d, _)) {
                alt (d.node) {
                    case (ast::decl_local(?loc)) {
                        add_name(values, d.span, loc.node.ident);
                    }
                    case (ast::decl_item(?it)) {
                        alt (it.node) {
                            case (ast::item_tag(?variants, _)) {
                                add_name(types, it.span, it.ident);
                                for (ast::variant v in variants) {
                                    add_name(values, v.span, v.node.name);
                                }
                            }
                            case (ast::item_const(_, _)) {
                                add_name(values, it.span, it.ident);
                            }
                            case (ast::item_fn(_, _)) {
                                add_name(values, it.span, it.ident);
                            }
                            case (ast::item_mod(_)) {
                                add_name(mods, it.span, it.ident);
                            }
                            case (ast::item_native_mod(_)) {
                                add_name(mods, it.span, it.ident);
                            }
                            case (ast::item_ty(_, _)) {
                                add_name(types, it.span, it.ident);
                            }
                            case (ast::item_res(_, _, _, _)) {
                                add_name(types, it.span, it.ident);
                                add_name(values, it.span, it.ident);
                            }
                            case (ast::item_obj(_, _, _)) {
                                add_name(types, it.span, it.ident);
                                add_name(values, it.span, it.ident);
                            }
                            case (_) { }
                        }
                    }
                }
            }
            case (_) { }
        }
    }
}

fn check_fn(&env e, &span sp, &ast::_fn f) {
    fn arg_name(&ast::arg a) -> ident { ret a.ident; }
    ensure_unique(e, sp, f.decl.inputs, arg_name, "argument");
}

type checker = @rec(mutable vec[ident] seen, str kind, session sess);

fn checker(&env e, str kind) -> checker {
    let vec[ident] seen = [];
    ret @rec(mutable seen=seen, kind=kind, sess=e.sess);
}

fn add_name(&checker ch, &span sp, &ident name) {
    for (ident s in ch.seen) {
        if (str::eq(s, name)) {
            ch.sess.span_fatal(sp, "duplicate " + ch.kind + " name: " + name);
        }
    }
    vec::push(ch.seen, name);
}

fn ident_id(&ident i) -> ident { ret i; }

fn ensure_unique[T](&env e, &span sp, &vec[T] elts, fn(&T) -> ident  id,
                    &str kind) {
    auto ch = checker(e, kind);
    for (T elt in elts) { add_name(ch, sp, id(elt)); }
}
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
