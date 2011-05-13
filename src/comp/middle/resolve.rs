import front::ast;
import front::ast::ident;
import front::ast::def;
import front::ast::def_id;
import front::ast::ann;
import front::creader;
import driver::session::session;
import util::common::new_def_hash;
import util::common::new_int_hash;
import util::common::new_uint_hash;
import util::common::new_str_hash;
import util::common::span;
import util::typestate_ann::ts_ann;
import std::map::hashmap;
import std::list::list;
import std::list::nil;
import std::list::cons;
import std::option;
import std::option::some;
import std::option::none;
import std::_str;
import std::_vec;

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
    scope_native_item(@ast::native_item);
    scope_loop(@ast::decl); // there's only 1 decl per loop.
    scope_block(ast::block);
    scope_arm(ast::arm);
}

tag import_state {
    todo(@ast::view_item, list[scope]);
    resolving(span);
    resolved(option::t[def] /* value */, option::t[def] /* type */);
}

type ext_hash = hashmap[tup(def_id,str),def];
fn new_ext_hash() -> ext_hash {
    fn hash(&tup(def_id,str) v) -> uint {
        ret _str::hash(v._1) + util::common::hash_def(v._0);
    }
    fn eq(&tup(def_id,str) v1, &tup(def_id,str) v2) -> bool {
        ret util::common::def_eq(v1._0, v2._0) &&
            _str::eq(v1._1, v2._1);
    }
    ret std::map::mk_hashmap[tup(def_id,str),def](hash, eq);
}

tag mod_index_entry {
    mie_view_item(@ast::view_item);
    mie_item(@ast::item);
    mie_tag_variant(@ast::item /* tag item */, uint /* variant index */);
}
type mod_index = hashmap[ident,list[mod_index_entry]];
type indexed_mod = rec(ast::_mod m, mod_index index);

tag nmod_index_entry {
    nmie_view_item(@ast::view_item);
    nmie_item(@ast::native_item);
}
type nmod_index = hashmap[ident,list[nmod_index_entry]];
type indexed_nmod = rec(ast::native_mod m, nmod_index index);

type def_map = hashmap[uint,def];

type env = rec(def_map def_map,
               hashmap[ast::def_num,import_state] imports,
               hashmap[ast::def_num,@indexed_mod] mod_map,
               hashmap[ast::def_num,@indexed_nmod] nmod_map,
               hashmap[def_id,vec[ident]] ext_map,
               ext_hash ext_cache,
               session sess);

// Used to distinguish between lookups from outside and from inside modules,
// since export restrictions should only be applied for the former.
tag dir { inside; outside; }

tag namespace {
    ns_value;
    ns_type;
}

fn resolve_crate(session sess, @ast::crate crate) -> def_map {
    auto e = @rec(def_map = new_uint_hash[def](),
                  imports = new_int_hash[import_state](),
                  mod_map = new_int_hash[@indexed_mod](),
                  nmod_map = new_int_hash[@indexed_nmod](),
                  ext_map = new_def_hash[vec[ident]](),
                  ext_cache = new_ext_hash(),
                  sess = sess);
    map_crate(e, *crate);
    resolve_imports(*e);
    resolve_names(e, *crate);
    ret e.def_map;
}

// Locate all modules and imports and index them, so that the next passes can
// resolve through them.

fn map_crate(&@env e, &ast::crate c) {
    auto cell = @mutable nil[scope];
    auto v = rec(visit_crate_pre = bind push_env_for_crate(cell, _),
                 visit_crate_post = bind pop_env_for_crate(cell, _),
                 visit_view_item_pre = bind visit_view_item(e, cell, _),
                 visit_item_pre = bind visit_item(e, cell, _),
                 visit_item_post = bind pop_env_for_item(cell, _)
                 with walk::default_visitor());
    // Register the top-level mod
    e.mod_map.insert(-1, @rec(m=c.node.module,
                              index=index_mod(c.node.module)));
    walk::walk_crate(v, c);

    fn visit_view_item(@env e, @mutable list[scope] sc, &@ast::view_item i) {
        alt (i.node) {
            case (ast::view_item_import(_, ?ids, ?defid)) {
                e.imports.insert(defid._1, todo(i, *sc));
            }
            case (_) {}
        }
    }
    fn visit_item(@env e, @mutable list[scope] sc, &@ast::item i) {
        push_env_for_item(sc, i);
        alt (i.node) {
            case (ast::item_mod(_, ?md, ?defid)) {
                auto index = index_mod(md);
                e.mod_map.insert(defid._1, @rec(m=md, index=index));
            }
            case (ast::item_native_mod(_, ?nmd, ?defid)) {
                auto index = index_nmod(nmd);
                e.nmod_map.insert(defid._1, @rec(m=nmd, index=index));
            }
            case (_) {}
        }
    }
}

fn resolve_imports(&env e) {
    for each (@tup(ast::def_num, import_state) it in e.imports.items()) {
        alt (it._1) {
            case (todo(?item, ?sc)) {
                resolve_import(e, item, sc);
            }
            case (resolved(_, _)) {}
        }
    }
}

fn resolve_names(&@env e, &ast::crate c) {
    auto cell = @mutable nil[scope];
    auto v = rec(visit_crate_pre = bind push_env_for_crate(cell, _),
                 visit_crate_post = bind pop_env_for_crate(cell, _),
                 visit_item_pre = bind push_env_for_item(cell, _),
                 visit_item_post = bind pop_env_for_item(cell, _),
                 visit_method_pre = bind push_env_for_method(cell, _),
                 visit_method_post = bind pop_env_for_method(cell, _),
                 visit_native_item_pre = bind push_env_for_n_item(cell, _),
                 visit_native_item_post = bind pop_env_for_n_item(cell, _),
                 visit_block_pre = bind push_env_for_block(cell, _),
                 visit_block_post = bind pop_env_for_block(cell, _),
                 visit_arm_pre = bind walk_arm(e, cell, _),
                 visit_arm_post = bind pop_env_for_arm(cell, _),
                 visit_expr_pre = bind walk_expr(e, cell, _),
                 visit_expr_post = bind pop_env_for_expr(cell, _),
                 visit_ty_pre = bind walk_ty(e, cell, _)
                 with walk::default_visitor());
    walk::walk_crate(v, c);

    fn walk_expr(@env e, @mutable list[scope] sc, &@ast::expr exp) {
        push_env_for_expr(sc, exp);
        alt (exp.node) {
            case (ast::expr_path(?p, ?a)) {
                auto df = lookup_path_strict(*e, *sc, exp.span, p.node.idents,
                                             ns_value);
                e.def_map.insert(ast::ann_tag(a), df);
            }
            case (_) {}
        }
    }
    fn walk_ty(@env e, @mutable list[scope] sc, &@ast::ty t) {
        alt (t.node) {
            case (ast::ty_path(?p, ?a)) {
                auto new_def = lookup_path_strict(*e, *sc, t.span,
                                                  p.node.idents, ns_type);
                e.def_map.insert(ast::ann_tag(a), new_def);
            }
            case (_) {}
        }
    }
    fn walk_arm(@env e, @mutable list[scope] sc, &ast::arm a) {
        walk_pat(*e, *sc, a.pat);
        push_env_for_arm(sc, a);
    }
    fn walk_pat(&env e, &list[scope] sc, &@ast::pat pat) {
        alt (pat.node) {
            case (ast::pat_tag(?p, ?children, ?a)) {
                auto fnd = lookup_path_strict(e, sc, p.span, p.node.idents,
                                              ns_value);
                alt (fnd) {
                    case (ast::def_variant(?did, ?vid)) {
                        e.def_map.insert(ast::ann_tag(a), fnd);
                    }
                    case (_) {
                        e.sess.span_err(p.span, "not a tag variant: " +
                                        _str::connect(p.node.idents, "::"));
                        fail;
                    }
                }
                for (@ast::pat child in children) {
                    walk_pat(e, sc, child);
                }
            }
            case (_) {}
        }
    }
}

// Helpers for tracking scope during a walk

fn push_env_for_crate(@mutable list[scope] sc, &ast::crate c) {
    *sc = cons[scope](scope_crate(@c), @*sc);
}
fn pop_env_for_crate(@mutable list[scope] sc, &ast::crate c) {
    *sc = std::list::cdr(*sc);
}

fn push_env_for_item(@mutable list[scope] sc, &@ast::item i) {
    *sc = cons[scope](scope_item(i), @*sc);
}
fn pop_env_for_item(@mutable list[scope] sc, &@ast::item i) {
    *sc = std::list::cdr(*sc);
}

fn push_env_for_method(@mutable list[scope] sc, &@ast::method m) {
    let vec[ast::ty_param] tp = vec();
    let @ast::item i = @rec(node=ast::item_fn(m.node.ident,
                                              m.node.meth,
                                              tp,
                                              m.node.id,
                                              m.node.ann),
                            span=m.span);
    *sc = cons[scope](scope_item(i), @*sc);
}
fn pop_env_for_method(@mutable list[scope] sc, &@ast::method m) {
    *sc = std::list::cdr(*sc);
}

fn push_env_for_n_item(@mutable list[scope] sc, &@ast::native_item i) {
    *sc = cons[scope](scope_native_item(i), @*sc);
}
fn pop_env_for_n_item(@mutable list[scope] sc, &@ast::native_item i) {
    *sc = std::list::cdr(*sc);
}

fn push_env_for_block(@mutable list[scope] sc, &ast::block b) {
    *sc = cons[scope](scope_block(b), @*sc);
}
fn pop_env_for_block(@mutable list[scope] sc, &ast::block b) {
    *sc = std::list::cdr(*sc);
}

fn push_env_for_expr(@mutable list[scope] sc, &@ast::expr x) {
    alt (x.node) {
        case (ast::expr_for(?d, _, _, _)) {
            *sc = cons[scope](scope_loop(d), @*sc);
        }
        case (ast::expr_for_each(?d, _, _, _)) {
            *sc = cons[scope](scope_loop(d), @*sc);
        }
        case (_) {}
    }
}
fn pop_env_for_expr(@mutable list[scope] sc, &@ast::expr x) {
    alt (x.node) {
        case (ast::expr_for(?d, _, _, _)) {
            *sc = std::list::cdr(*sc);
        }
        case (ast::expr_for_each(?d, _, _, _)) {
            *sc = std::list::cdr(*sc);
        }
        case (_) {}
    }
}

fn push_env_for_arm(@mutable list[scope] sc, &ast::arm p) {
    *sc = cons[scope](scope_arm(p), @*sc);
}
fn pop_env_for_arm(@mutable list[scope] sc, &ast::arm p) {
    *sc = std::list::cdr(*sc);
}

// Import resolution

fn resolve_import(&env e, &@ast::view_item it, &list[scope] sc) {
    auto defid; auto ids;
    alt (it.node) {
        case (ast::view_item_import(_, ?_ids, ?_defid)) {
            defid = _defid; ids = _ids;
        }
    }
    e.imports.insert(defid._1, resolving(it.span));
    
    auto n_idents = _vec::len(ids);
    auto end_id = ids.(n_idents - 1u);

    if (n_idents == 1u) {
        register(e, defid, it.span, end_id,
                 lookup_in_scope(e, sc, it.span, end_id, ns_value),
                 lookup_in_scope(e, sc, it.span, end_id, ns_type));
    } else {
        auto dcur = lookup_in_scope_strict(e, sc, it.span, ids.(0), ns_value);
        auto i = 1u;
        while (true) {
            if (!is_module(dcur)) {
                e.sess.span_err(it.span, ids.(i-1u) +
                                " is not a module or crate");
            }
            if (i == n_idents - 1u) {
                register(e, defid, it.span, end_id,
                         lookup_in_mod(e, dcur, end_id, ns_value, outside),
                         lookup_in_mod(e, dcur, end_id, ns_type, outside));
                break;
            } else {
                dcur = lookup_in_mod_strict(e, dcur, it.span, ids.(i),
                                            ns_value, outside);
                i += 1u;
            }
        }
    }

    fn register(&env e, def_id defid, &span sp, &ident id,
                &option::t[def] val, &option::t[def] typ) {
        if (option::is_none(val) && option::is_none(typ)) {
            unresolved(e, sp, id, "import");
        }
        e.imports.insert(defid._1, resolved(val, typ));
    }
}

// Utilities

fn is_module(def d) -> bool {
    alt (d) {
        case (ast::def_mod(_)) { ret true; }
        case (ast::def_native_mod(_)) { ret true; }
        case (_) { ret false; }
    }
}

fn ns_name(namespace ns) -> str {
    alt (ns) {
        case (ns_type) { ret "typename"; }
        case (ns_value) { ret "name"; }
    }
}

fn unresolved(&env e, &span sp, &ident id, &str kind) {
    e.sess.span_err(sp, "unresolved " + kind + ": " + id);
}

// Lookup helpers

fn lookup_path_strict(&env e, &list[scope] sc, &span sp, vec[ident] idents,
                      namespace ns) -> def {
    auto n_idents = _vec::len(idents);
    auto dcur = lookup_in_scope_strict(e, sc, sp, idents.(0), ns);
    auto i = 1u;
    while (i < n_idents) {
        if (!is_module(dcur)) {
            e.sess.span_err(sp, idents.(i-1u) +
                            " can't be dereferenced as a module");
        }
        dcur = lookup_in_mod_strict(e, dcur, sp, idents.(i), ns, outside);
        i += 1u;
    }
    if (is_module(dcur)) {
        e.sess.span_err(sp, _str::connect(idents, "::") +
                        " is a module, not a " + ns_name(ns));
    }
    ret dcur;
}
                      
fn lookup_in_scope_strict(&env e, list[scope] sc, &span sp, &ident id,
                        namespace ns) -> def {
    alt (lookup_in_scope(e, sc, sp, id, ns)) {
        case (none[def]) {
            unresolved(e, sp, id, ns_name(ns));
            fail;
        }
        case (some[def](?d)) {
            ret d;
        }
    }
}

fn scope_is_fn(&scope sc) -> bool {
    ret alt (sc) {
        case (scope_item(?it)) {
            alt (it.node) {
                case (ast::item_fn(_, _, _, _, _)) { true }
                case (_) { false }
            }
        }
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
    ret alt (d) {
        case (ast::def_obj_field(_)) { true }
        case (_) { false }
    };
}

fn lookup_in_scope(&env e, list[scope] sc, &span sp, &ident id, namespace ns)
    -> option::t[def] {
    fn in_scope(&env e, &ident id, &scope s, namespace ns)
        -> option::t[def] {
        alt (s) {
            case (scope_crate(?c)) {
                auto defid = tup(ast::local_crate, -1);
                ret lookup_in_regular_mod(e, defid, id, ns, inside);
            }
            case (scope_item(?it)) {
                alt (it.node) {
                    case (ast::item_fn(_, ?f, ?ty_params, _, _)) {
                        ret lookup_in_fn(id, f.decl, ty_params, ns);
                    }
                    case (ast::item_obj(_, ?ob, ?ty_params, _, _)) {
                        ret lookup_in_obj(id, ob, ty_params, ns);
                    }
                    case (ast::item_tag(_, _, ?ty_params, _, _)) {
                        if (ns == ns_type) {
                            ret lookup_in_ty_params(id, ty_params);
                        }
                    }
                    case (ast::item_mod(_, _, ?defid)) {
                        ret lookup_in_regular_mod(e, defid, id, ns, inside);
                    }
                    case (ast::item_native_mod(_, ?m, ?defid)) {
                        ret lookup_in_native_mod(e, defid, id, ns);
                    }
                    case (ast::item_ty(_, _, ?ty_params, _, _)) {
                        if (ns == ns_type) {
                            ret lookup_in_ty_params(id, ty_params);
                        }
                    }
                    case (_) {}
                }
            }
            case (scope_native_item(?it)) {
                alt (it.node) {
                    case (ast::native_item_fn(_, _, ?decl, ?ty_params, _, _)){
                        ret lookup_in_fn(id, decl, ty_params, ns);
                    }
                }
            }
            case (scope_loop(?d)) {
                if (ns == ns_value) {
                    alt (d.node) {
                        case (ast::decl_local(?local)) {
                            if (_str::eq(local.ident, id)) {
                                ret some(ast::def_local(local.id));
                            }
                        }
                    }
                }
            }
            case (scope_block(?b)) {
                ret lookup_in_block(id, b.node, ns);
            }
            case (scope_arm(?a)) {
                if (ns == ns_value) {
                    ret lookup_in_pat(id, *a.pat);
                }
            }
        }
        ret none[def];
    }

    auto left_fn = false;
    // Used to determine whether obj fields are in scope
    auto left_fn_level2 = false;
    while (true) {
        alt (sc) {
            case (nil[scope]) {
                ret none[def];
            }
            case (cons[scope](?hd, ?tl)) {
                auto fnd = in_scope(e, id, hd, ns);
                if (fnd != none[def]) {
                    auto df = option::get(fnd);
                    if ((left_fn && def_is_local(df)) ||
                        (left_fn_level2 && def_is_obj_field(df))) {
                        e.sess.span_err
                            (sp, "attempted dynamic environment-capture");
                    }
                    ret fnd;
                }
                if (left_fn) { left_fn_level2 = true; }
                if (ns == ns_value && !left_fn) {
                    left_fn = scope_is_fn(hd);
                }
                sc = *tl;
            }
        }
    }
}

fn lookup_in_ty_params(&ident id, &vec[ast::ty_param] ty_params)
    -> option::t[def] {
    auto i = 0u;
    for (ast::ty_param tp in ty_params) {
        if (_str::eq(tp, id)) {
            ret some(ast::def_ty_arg(i));
        }
        i += 1u;
    }
    ret none[def];
}

fn lookup_in_pat(&ident id, &ast::pat pat) -> option::t[def] {
    alt (pat.node) {
        case (ast::pat_bind(?name, ?defid, _)) {
            if (_str::eq(name, id)) { ret some(ast::def_binding(defid)); }
        }
        case (ast::pat_wild(_)) {}
        case (ast::pat_lit(_, _)) {}
        case (ast::pat_tag(_, ?pats, _)) {
            for (@ast::pat p in pats) {
                auto found = lookup_in_pat(id, *p);
                if (found != none[def]) { ret found; }
            }
        }
    }
    ret none[def];
}


fn lookup_in_fn(&ident id, &ast::fn_decl decl, &vec[ast::ty_param] ty_params,
                namespace ns) -> option::t[def] {
    if (ns == ns_value) {
        for (ast::arg a in decl.inputs) {
            if (_str::eq(a.ident, id)) {
                ret some(ast::def_arg(a.id));
            }
        }
        ret none[def];
    } else {
        ret lookup_in_ty_params(id, ty_params);
    }
}

fn lookup_in_obj(&ident id, &ast::_obj ob, &vec[ast::ty_param] ty_params,
                 namespace ns) -> option::t[def] {
    if (ns == ns_value) {
        for (ast::obj_field f in ob.fields) {
            if (_str::eq(f.ident, id)) {
                ret some(ast::def_obj_field(f.id));
            }
        }
        ret none[def];
    } else {
        ret lookup_in_ty_params(id, ty_params);
    }
}

fn lookup_in_block(&ident id, &ast::block_ b, namespace ns)
    -> option::t[def] {
    for (@ast::stmt st in b.stmts) {
        alt (st.node) {
            case (ast::stmt_decl(?d,_)) {
                alt (d.node) {
                    case (ast::decl_local(?loc)) {
                        if (ns == ns_value && _str::eq(id, loc.ident)) {
                            ret some(ast::def_local(loc.id));
                        }
                    }
                    case (ast::decl_item(?it)) {
                        alt (it.node) {
                            case (ast::item_tag(?name, ?variants, _,
                                               ?defid, _)) {
                                if (ns == ns_type) {
                                    if (_str::eq(name, id)) {
                                        ret some(ast::def_ty(defid));
                                    }
                                } else {
                                    for (ast::variant v in variants) {
                                        if (_str::eq(v.node.name, id)) {
                                            ret some(ast::def_variant(
                                                      defid, v.node.id));
                                        }
                                    }
                                }
                            }
                            case (_) {
                                if (_str::eq(ast::item_ident(it), id)) {
                                    auto found = found_def_item(it, ns);
                                    if (found != none[def]) { ret found; }
                                }
                            }
                        }
                    }
                }
            }
            case (_) {}
        }
    }
    ret none[def];
}

fn found_def_item(@ast::item i, namespace ns) -> option::t[def] {
    alt (i.node) {
        case (ast::item_const(_, _, _, ?defid, _)) {
            if (ns == ns_value) { ret some(ast::def_const(defid)); }
        }
        case (ast::item_fn(_, _, _, ?defid, _)) {
            if (ns == ns_value) { ret some(ast::def_fn(defid)); }
        }
        case (ast::item_mod(_, _, ?defid)) {
            ret some(ast::def_mod(defid));
        }
        case (ast::item_native_mod(_, _, ?defid)) {
            ret some(ast::def_native_mod(defid));
        }
        case (ast::item_ty(_, _, _, ?defid, _)) {
            if (ns == ns_type) { ret some(ast::def_ty(defid)); }
        }
        case (ast::item_tag(_, _, _, ?defid, _)) {
            if (ns == ns_type) { ret some(ast::def_ty(defid)); }
        }
        case (ast::item_obj(_, _, _, ?odid, _)) {
            if (ns == ns_value) { ret some(ast::def_obj(odid.ctor)); }
            else { ret some(ast::def_obj(odid.ty)); }
        }
        case (_) { }
    }
    ret none[def];
}

fn lookup_in_mod_strict(&env e, def m, &span sp, &ident id,
                        namespace ns, dir dr) -> def {
    alt (lookup_in_mod(e, m, id, ns, dr)) {
        case (none[def]) {
            unresolved(e, sp, id, ns_name(ns));
            fail;
        }
        case (some[def](?d)) {
            ret d;
        }
    }
}

fn lookup_in_mod(&env e, def m, &ident id, namespace ns, dir dr)
    -> option::t[def] {
    auto defid = ast::def_id_of_def(m);
    if (defid._0 != ast::local_crate) { // Not in this crate
        auto cached = e.ext_cache.find(tup(defid,id));
        if (cached != none[def] && check_def_by_ns(option::get(cached), ns)) {
            ret cached;
        }
        auto path = vec(id);
        if (defid._1 != -1) {
            path = e.ext_map.get(defid) + path;
        }
        auto fnd = lookup_external(e, defid._0, path, ns);
        if (fnd != none[def]) {
            e.ext_cache.insert(tup(defid,id), option::get(fnd));
        }
        ret fnd;
    }
    alt (m) {
        case (ast::def_mod(?defid)) {
            ret lookup_in_regular_mod(e, defid, id, ns, dr);
        }
        case (ast::def_native_mod(?defid)) {
            ret lookup_in_native_mod(e, defid, id, ns);
        }
    }
}

fn found_view_item(&env e, @ast::view_item vi, namespace ns)
    -> option::t[def] {
    alt (vi.node) {
        case (ast::view_item_use(_, _, _, ?cnum)) {
            ret some(ast::def_mod(tup(option::get(cnum), -1)));
        }
        case (ast::view_item_import(_, _, ?defid)) {
            ret lookup_import(e, defid, ns);
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
        }
        case (resolved(?val, ?typ)) {
            ret alt (ns) { case (ns_value) { val }
                           case (ns_type) { typ } };
        }
    }
}

fn lookup_in_regular_mod(&env e, def_id defid, &ident id, namespace ns, dir dr)
    -> option::t[def] {
    auto info = e.mod_map.get(defid._1);
    auto found = info.index.find(id);
    if (option::is_none(found) || 
        (dr == outside && !ast::is_exported(id, info.m))) {
        ret none[def];
    }
    auto lst = option::get(found);
    while (true) {
        alt (lst) {
            case (nil[mod_index_entry]) {
                ret none[def];
            }
            case (cons[mod_index_entry](?hd, ?tl)) {
                auto found = lookup_in_mie(e, hd, ns);
                if (!option::is_none(found)) { ret found; }
                lst = *tl;
            }
        }
    }
}

fn lookup_in_mie(&env e, &mod_index_entry mie, namespace ns)
    -> option::t[def] {
    alt (mie) {
        case (mie_view_item(?view_item)) {
            ret found_view_item(e, view_item, ns);
        }
        case (mie_item(?item)) {
            ret found_def_item(item, ns);
        }
        case (mie_tag_variant(?item, ?variant_idx)) {
            alt (item.node) {
                case (ast::item_tag(_, ?variants, _, ?tid, _)) {
                    if (ns == ns_value) {
                        auto vid = variants.(variant_idx).node.id;
                        ret some(ast::def_variant(tid, vid));
                    } else {
                        ret none[def];
                    }
                }
            }
        }
    }
}

fn lookup_in_native_mod(&env e, def_id defid, &ident id, namespace ns)
    -> option::t[def] {
    auto info = e.nmod_map.get(defid._1);
    auto found = info.index.find(id);
    if (option::is_none(found)) {
        ret none[def];
    }
    auto lst = option::get(found);
    while (true) {
        alt (lst) {
            case (nil[nmod_index_entry]) {
                ret none[def];
            }
            case (cons[nmod_index_entry](?hd, ?tl)) {
                auto found = lookup_in_nmie(e, hd, ns);
                if (!option::is_none(found)) { ret found; }
                lst = *tl;
            }
        }
    }
}
    
fn lookup_in_nmie(&env e, &nmod_index_entry nmie, namespace ns)
    -> option::t[def] {
    alt (nmie) {
        case (nmie_view_item(?view_item)) {
            ret found_view_item(e, view_item, ns);
        }
        case (nmie_item(?item)) {
            alt (item.node) {
                case (ast::native_item_ty(_, ?id)) {
                    if (ns == ns_type) {
                        ret some(ast::def_native_ty(id));
                    }
                }
                case (ast::native_item_fn(_, _, _, _, ?id, _)) {
                    if (ns == ns_value) {
                        ret some(ast::def_native_fn(id));
                    }
                }
            }
        }
        case (_) {}
    }
    ret none[def];
}

// Module indexing

fn add_to_index[T](&hashmap[ident,list[T]] index, &ident id, &T ent) {
    alt (index.find(id)) {
        case (none[list[T]]) {
            index.insert(id, cons(ent, @nil[T]));
        }
        case (some[list[T]](?prev)) {
            index.insert(id, cons(ent, @prev));
        }
    }
}

fn index_mod(&ast::_mod md) -> mod_index {
    auto index = new_str_hash[list[mod_index_entry]]();

    for (@ast::view_item it in md.view_items) {
        alt (it.node) {
            case(ast::view_item_use(?id, _, _, _)) {
                add_to_index(index, id, mie_view_item(it));
            }
            case(ast::view_item_import(?def_ident,_,_)) {
                add_to_index(index, def_ident, mie_view_item(it));
            }
            case(ast::view_item_export(_)) {}
        }
    }

    for (@ast::item it in md.items) {
        alt (it.node) {
            case (ast::item_const(?id, _, _, _, _)) {
                add_to_index(index, id, mie_item(it));
            }
            case (ast::item_fn(?id, _, _, _, _)) {
                add_to_index(index, id, mie_item(it));
            }
            case (ast::item_mod(?id, _, _)) {
                add_to_index(index, id, mie_item(it));
            }
            case (ast::item_native_mod(?id, _, _)) {
                add_to_index(index, id, mie_item(it));
            }
            case (ast::item_ty(?id, _, _, _, _)) {
                add_to_index(index, id, mie_item(it));
            }
            case (ast::item_tag(?id, ?variants, _, _, _)) {
                add_to_index(index, id, mie_item(it));
                let uint variant_idx = 0u;
                for (ast::variant v in variants) {
                    add_to_index(index, v.node.name,
                                     mie_tag_variant(it, variant_idx));
                    variant_idx += 1u;
                }
            }
            case (ast::item_obj(?id, _, _, _, _)) {
                add_to_index(index, id, mie_item(it));
            }
        }
    }

    ret index;
}

fn index_nmod(&ast::native_mod md) -> nmod_index {
    auto index = new_str_hash[list[nmod_index_entry]]();

    for (@ast::view_item it in md.view_items) {
        alt (it.node) {
            case(ast::view_item_import(?def_ident,_,_)) {
                add_to_index(index, def_ident, nmie_view_item(it));
            }
            case(ast::view_item_export(_)) {}
        }
    }

    for (@ast::native_item it in md.items) {
        alt (it.node) {
            case (ast::native_item_ty(?id, _)) {
                add_to_index(index, id, nmie_item(it));
            }
            case (ast::native_item_fn(?id, _, _, _, _, _)) {
                add_to_index(index, id, nmie_item(it));
            }
        }
    }

    ret index;
}

// External lookups

// FIXME creader should handle multiple namespaces
fn check_def_by_ns(def d, namespace ns) -> bool {
    ret alt (d) {
        case (ast::def_fn(?id)) { ns == ns_value }
        case (ast::def_obj(?id)) { ns == ns_value }
        case (ast::def_obj_field(?id)) { ns == ns_value }
        case (ast::def_mod(?id)) { true }
        case (ast::def_native_mod(?id)) { true }
        case (ast::def_const(?id)) { ns == ns_value }
        case (ast::def_arg(?id)) { ns == ns_value }
        case (ast::def_local(?id)) { ns == ns_value }
        case (ast::def_variant(_, ?id)) { ns == ns_value }
        case (ast::def_ty(?id)) { ns == ns_type }
        case (ast::def_binding(?id)) { ns == ns_type }
        case (ast::def_use(?id)) { true }
        case (ast::def_native_ty(?id)) { ns == ns_type }
        case (ast::def_native_fn(?id)) { ns == ns_value }
    };
}

fn lookup_external(&env e, int cnum, vec[ident] ids, namespace ns)
    -> option::t[def] {
    auto found = creader::lookup_def(e.sess, cnum, ids);
    if (found != none[def]) {
        auto d = option::get(found);
        if (!check_def_by_ns(d, ns)) { ret none[def]; }
        e.ext_map.insert(ast::def_id_of_def(d), ids);
    }
    ret found;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
