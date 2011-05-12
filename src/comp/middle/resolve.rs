import front.ast;
import front.ast.ident;
import front.ast.def;
import front.ast.def_id;
import front.ast.ann;
import front.creader;
import driver.session.session;
import util.common.new_def_hash;
import util.common.new_int_hash;
import util.common.new_uint_hash;
import util.common.new_str_hash;
import util.common.span;
import util.typestate_ann.ts_ann;
import std.Map.hashmap;
import std.List;
import std.List.list;
import std.List.nil;
import std.List.cons;
import std.Option;
import std.Option.some;
import std.Option.none;
import std.Str;
import std.Vec;

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
    scope_crate(@ast.crate);
    scope_item(@ast.item);
    scope_native_item(@ast.native_item);
    scope_loop(@ast.decl); // there's only 1 decl per loop.
    scope_block(ast.block);
    scope_arm(ast.arm);
}

tag import_state {
    todo(@ast.view_item, list[scope]);
    resolving(span);
    resolved(Option.t[def] /* value */, Option.t[def] /* type */);
}

type ext_hash = hashmap[tup(def_id,str),def];
fn new_ext_hash() -> ext_hash {
    fn hash(&tup(def_id,str) v) -> uint {
        ret Str.hash(v._1) + util.common.hash_def(v._0);
    }
    fn eq(&tup(def_id,str) v1, &tup(def_id,str) v2) -> bool {
        ret util.common.def_eq(v1._0, v2._0) &&
            Str.eq(v1._1, v2._1);
    }
    ret std.Map.mk_hashmap[tup(def_id,str),def](hash, eq);
}

tag mod_index_entry {
    mie_view_item(@ast.view_item);
    mie_item(@ast.item);
    mie_tag_variant(@ast.item /* tag item */, uint /* variant index */);
}
type mod_index = hashmap[ident,mod_index_entry];
type indexed_mod = rec(ast._mod m, mod_index index);

tag native_mod_index_entry {
    nmie_view_item(@ast.view_item);
    nmie_item(@ast.native_item);
}
type nmod_index = hashmap[ident,native_mod_index_entry];
type indexed_nmod = rec(ast.native_mod m, nmod_index index);

type def_map = hashmap[uint,def];

type env = rec(def_map def_map,
               hashmap[ast.def_num,import_state] imports,
               hashmap[ast.def_num,@indexed_mod] mod_map,
               hashmap[ast.def_num,@indexed_nmod] nmod_map,
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

fn resolve_crate(session sess, @ast.crate crate)
    -> tup(@ast.crate, def_map) {
    auto e = @rec(def_map = new_uint_hash[def](),
                  imports = new_int_hash[import_state](),
                  mod_map = new_int_hash[@indexed_mod](),
                  nmod_map = new_int_hash[@indexed_nmod](),
                  ext_map = new_def_hash[vec[ident]](),
                  ext_cache = new_ext_hash(),
                  sess = sess);
    map_crate(e, *crate);
    resolve_imports(*e);
    ret tup(resolve_names(e, *crate), e.def_map);
}

// Locate all modules and imports and index them, so that the next passes can
// resolve through them.

fn map_crate(&@env e, &ast.crate c) {
    auto cell = @mutable nil[scope];
    auto v = rec(visit_crate_pre = bind push_env_for_crate(cell, _),
                 visit_crate_post = bind pop_env_for_crate(cell, _),
                 visit_view_item_pre = bind visit_view_item(e, cell, _),
                 visit_item_pre = bind push_env_for_item_map_mod(e, cell, _),
                 visit_item_post = bind pop_env_for_item(cell, _)
                 with walk.default_visitor());
    // Register the top-level mod
    e.mod_map.insert(-1, @rec(m=c.node.module,
                              index=index_mod(c.node.module)));
    walk.walk_crate(v, c);

    // Helpers for this pass.
    fn push_env_for_crate(@mutable list[scope] sc, &ast.crate c) {
        *sc = cons[scope](scope_crate(@c), @*sc);
    }
    fn pop_env_for_crate(@mutable list[scope] sc, &ast.crate c) {
        *sc = List.cdr(*sc);
    }
    fn push_env_for_item_map_mod(@env e, @mutable list[scope] sc,
                                 &@ast.item i) {
        *sc = cons[scope](scope_item(i), @*sc);
        alt (i.node) {
            case (ast.item_mod(_, ?md, ?defid)) {
                auto index = index_mod(md);
                e.mod_map.insert(defid._1, @rec(m=md, index=index));
            }
            case (ast.item_native_mod(_, ?nmd, ?defid)) {
                auto index = index_nmod(nmd);
                e.nmod_map.insert(defid._1, @rec(m=nmd, index=index));
            }
            case (_) {}
        }
    }
    fn pop_env_for_item(@mutable list[scope] sc, &@ast.item i) {
        *sc = List.cdr(*sc);
    }
    fn visit_view_item(@env e, @mutable list[scope] sc, &@ast.view_item i) {
        alt (i.node) {
            case (ast.view_item_import(_, ?ids, ?defid)) {
                e.imports.insert(defid._1, todo(i, *sc));
            }
            case (_) {}
        }
    }
}

fn resolve_imports(&env e) {
    for each (@tup(ast.def_num, import_state) it in e.imports.items()) {
        alt (it._1) {
            case (todo(?item, ?sc)) {
                resolve_import(e, item, sc);
            }
            case (resolved(_, _)) {}
        }
    }
}

// FIXME this should use walk (will need to add walk_arm)
fn resolve_names(&@env e, &ast.crate c) -> @ast.crate {
    auto fld = @rec(fold_pat_tag = bind fold_pat_tag(e,_,_,_,_,_),
                    fold_expr_path = bind fold_expr_path(e,_,_,_,_),
                    fold_ty_path = bind fold_ty_path(e,_,_,_,_),
                    update_env_for_crate = bind update_env_for_crate(_,_),
                    update_env_for_item = bind update_env_for_item(_,_),
                    update_env_for_native_item =
                       bind update_env_for_native_item(_,_),
                    update_env_for_block = bind update_env_for_block(_,_),
                    update_env_for_arm = bind update_env_for_arm(_,_),
                    update_env_for_expr = bind update_env_for_expr(_,_)
                    with *fold.new_identity_fold[list[scope]]());
    ret fold.fold_crate(nil[scope], fld, @c);

    // Helpers for this pass

    fn update_env_for_crate(&list[scope] sc, &@ast.crate c) -> list[scope] {
        ret cons[scope](scope_crate(c), @sc);
    }
    fn update_env_for_item(&list[scope] sc, &@ast.item i) -> list[scope] {
        ret cons[scope](scope_item(i), @sc);
    }
    fn update_env_for_native_item(&list[scope] sc, &@ast.native_item i)
        -> list[scope] {
        ret cons[scope](scope_native_item(i), @sc);
    }
    fn update_env_for_block(&list[scope] sc, &ast.block b) -> list[scope] {
        ret cons[scope](scope_block(b), @sc);
    }
    fn update_env_for_expr(&list[scope] sc, &@ast.expr x) -> list[scope] {
        alt (x.node) {
            case (ast.expr_for(?d, _, _, _)) {
                ret cons[scope](scope_loop(d), @sc);
            }
            case (ast.expr_for_each(?d, _, _, _)) {
                ret cons[scope](scope_loop(d), @sc);
            }
            case (_) { ret sc; }
        }
    }
    fn update_env_for_arm(&list[scope] sc, &ast.arm p) -> list[scope] {
        ret cons[scope](scope_arm(p), @sc);
    }
}

fn lookup_import(&env e, def_id defid, namespace ns) -> Option.t[def] {
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

fn resolve_import(&env e, &@ast.view_item it, &list[scope] sc) {
    auto defid; auto ids;
    alt (it.node) {
        case (ast.view_item_import(_, ?_ids, ?_defid)) {
            defid = _defid; ids = _ids;
        }
    }
    e.imports.insert(defid._1, resolving(it.span));
    
    auto n_idents = Vec.len(ids);
    auto end_id = ids.(n_idents - 1u);

    if (n_idents == 1u) {
        register(e, defid, it.span, end_id,
                 lookup_in_scope(e, sc, end_id, ns_value),
                 lookup_in_scope(e, sc, end_id, ns_type));
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

    fn register(&env e, def_id defid, &span sp, ident id,
                Option.t[def] val, Option.t[def] typ) {
        if (val == none[def] && typ == none[def]) {
            unresolved(e, sp, id, "import");
        }
        e.imports.insert(defid._1, resolved(val, typ));
    }
}

fn fold_expr_path(@env e, &list[scope] sc, &span sp, &ast.path p, &ann a)
    -> @ast.expr {
    auto df = lookup_path_strict(*e, sc, sp, p.node.idents, ns_value);
    e.def_map.insert(ast.ann_tag(a), df);
    ret @fold.respan(sp, ast.expr_path(p, a));
}


fn fold_pat_tag(@env e, &list[scope] sc, &span sp, &ast.path p,
                &vec[@ast.pat] args, &ann a) -> @ast.pat {
    alt (lookup_path_strict(*e, sc, sp, p.node.idents, ns_value)) {
        case (ast.def_variant(?did, ?vid)) {
            e.def_map.insert(ast.ann_tag(a), ast.def_variant(did, vid));
            ret @fold.respan[ast.pat_](sp, ast.pat_tag(p, args, a));
        }
        case (_) {
            e.sess.span_err(sp, "not a tag variant: " +
                            Str.connect(p.node.idents, ":"));
            fail;
        }
    }
}

fn fold_ty_path(@env e, &list[scope] sc, &span sp, &ast.path p,
                &ast.ann a) -> @ast.ty {
    auto new_def = lookup_path_strict(*e, sc, sp, p.node.idents, ns_type);
    e.def_map.insert(ast.ann_tag(a), new_def);
    ret @fold.respan[ast.ty_](sp, ast.ty_path(p, a));
}

fn is_module(def d) -> bool {
    alt (d) {
        case (ast.def_mod(_)) { ret true; }
        case (ast.def_native_mod(_)) { ret true; }
        case (_) { ret false; }
    }
}

fn ns_name(namespace ns) -> str {
    alt (ns) {
        case (ns_type) { ret "typename"; }
        case (ns_value) { ret "name"; }
    }
}

fn unresolved(&env e, &span sp, ident id, str kind) {
    e.sess.span_err(sp, "unresolved " + kind + ": " + id);
}

fn lookup_path_strict(&env e, &list[scope] sc, &span sp, vec[ident] idents,
                      namespace ns) -> def {
    auto n_idents = Vec.len(idents);
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
        e.sess.span_err(sp, Str.connect(idents, ":") +
                        " is a module, not a " + ns_name(ns));
    }
    ret dcur;
}
                      
fn lookup_in_scope_strict(&env e, list[scope] sc, &span sp, ident id,
                        namespace ns) -> def {
    alt (lookup_in_scope(e, sc, id, ns)) {
        case (none[def]) {
            unresolved(e, sp, id, ns_name(ns));
            fail;
        }
        case (some[def](?d)) {
            ret d;
        }
    }
}

fn lookup_in_scope(&env e, list[scope] sc, ident id, namespace ns)
    -> Option.t[def] {
    fn in_scope(&env e, ident id, &scope s, namespace ns)
        -> Option.t[def] {
        alt (s) {
            case (scope_crate(?c)) {
                auto defid = tup(ast.local_crate, -1);
                ret lookup_in_regular_mod(e, defid, id, ns, inside);
            }
            case (scope_item(?it)) {
                alt (it.node) {
                    case (ast.item_fn(_, ?f, ?ty_params, _, _)) {
                        ret lookup_in_fn(id, f.decl, ty_params, ns);
                    }
                    case (ast.item_obj(_, ?ob, ?ty_params, _, _)) {
                        ret lookup_in_obj(id, ob, ty_params, ns);
                    }
                    case (ast.item_tag(_, _, ?ty_params, _, _)) {
                        if (ns == ns_type) {
                            ret lookup_in_ty_params(id, ty_params);
                        }
                    }
                    case (ast.item_mod(_, _, ?defid)) {
                        ret lookup_in_regular_mod(e, defid, id, ns, inside);
                    }
                    case (ast.item_native_mod(_, ?m, ?defid)) {
                        ret lookup_in_native_mod(e, defid, id, ns);
                    }
                    case (ast.item_ty(_, _, ?ty_params, _, _)) {
                        if (ns == ns_type) {
                            ret lookup_in_ty_params(id, ty_params);
                        }
                    }
                    case (_) {}
                }
            }

            case (scope_native_item(?it)) {
                alt (it.node) {
                    case (ast.native_item_fn(_, _, ?decl, ?ty_params, _, _)) {
                        ret lookup_in_fn(id, decl, ty_params, ns);
                    }
                }
            }

            case (scope_loop(?d)) {
                if (ns == ns_value) {
                    alt (d.node) {
                        case (ast.decl_local(?local)) {
                            if (Str.eq(local.ident, id)) {
                                ret some(ast.def_local(local.id));
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

    while (true) {
        alt (sc) {
            case (nil[scope]) {
                ret none[def];
            }
            case (cons[scope](?hd, ?tl)) {
                alt (in_scope(e, id, hd, ns)) {
                    case (some[def](?x)) { ret some(x); }
                    case (_) { sc = *tl; }
                }
            }
        }
    }
}

fn lookup_in_ty_params(ident id, &vec[ast.ty_param] ty_params)
    -> Option.t[def] {
    auto i = 0u;
    for (ast.ty_param tp in ty_params) {
        if (Str.eq(tp, id)) {
            ret some(ast.def_ty_arg(i));
        }
        i += 1u;
    }
    ret none[def];
}

fn lookup_in_pat(ident id, &ast.pat pat) -> Option.t[def] {
    alt (pat.node) {
        case (ast.pat_bind(?name, ?defid, _)) {
            if (Str.eq(name, id)) { ret some(ast.def_binding(defid)); }
        }
        case (ast.pat_wild(_)) {}
        case (ast.pat_lit(_, _)) {}
        case (ast.pat_tag(_, ?pats, _)) {
            for (@ast.pat p in pats) {
                auto found = lookup_in_pat(id, *p);
                if (found != none[def]) { ret found; }
            }
        }
    }
    ret none[def];
}


fn lookup_in_fn(ident id, &ast.fn_decl decl,
                &vec[ast.ty_param] ty_params, namespace ns) -> Option.t[def] {
    if (ns == ns_value) {
        for (ast.arg a in decl.inputs) {
            if (Str.eq(a.ident, id)) {
                ret some(ast.def_arg(a.id));
            }
        }
        ret none[def];
    } else {
        ret lookup_in_ty_params(id, ty_params);
    }
}

fn lookup_in_obj(ident id, &ast._obj ob, &vec[ast.ty_param] ty_params,
                 namespace ns) -> Option.t[def] {
    if (ns == ns_value) {
        for (ast.obj_field f in ob.fields) {
            if (Str.eq(f.ident, id)) {
                ret some(ast.def_obj_field(f.id));
            }
        }
        ret none[def];
    } else {
        ret lookup_in_ty_params(id, ty_params);
    }
}

fn lookup_in_block(ident id, &ast.block_ b, namespace ns)
    -> Option.t[def] {
    for (@ast.stmt st in b.stmts) {
        alt (st.node) {
            case (ast.stmt_decl(?d,_)) {
                alt (d.node) {
                    case (ast.decl_local(?loc)) {
                        if (ns == ns_value && Str.eq(id, loc.ident)) {
                            ret some(ast.def_local(loc.id));
                        }
                    }
                    case (ast.decl_item(?it)) {
                        alt (it.node) {
                            case (ast.item_tag(?name, ?variants, _,
                                               ?defid, _)) {
                                if (ns == ns_type) {
                                    if (Str.eq(name, id)) {
                                        ret some(ast.def_ty(defid));
                                    }
                                } else {
                                    for (ast.variant v in variants) {
                                        if (Str.eq(v.node.name, id)) {
                                            ret some(ast.def_variant(
                                                      defid, v.node.id));
                                        }
                                    }
                                }
                            }
                            case (_) {
                                if (Str.eq(ast.item_ident(it), id)) {
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

fn found_def_item(@ast.item i, namespace ns) -> Option.t[def] {
    alt (i.node) {
        case (ast.item_const(_, _, _, ?defid, _)) {
            if (ns == ns_value) { ret some(ast.def_const(defid)); }
        }
        case (ast.item_fn(_, _, _, ?defid, _)) {
            if (ns == ns_value) { ret some(ast.def_fn(defid)); }
        }
        case (ast.item_mod(_, _, ?defid)) {
            ret some(ast.def_mod(defid));
        }
        case (ast.item_native_mod(_, _, ?defid)) {
            ret some(ast.def_native_mod(defid));
        }
        case (ast.item_ty(_, _, _, ?defid, _)) {
            if (ns == ns_type) { ret some(ast.def_ty(defid)); }
        }
        case (ast.item_tag(_, _, _, ?defid, _)) {
            if (ns == ns_type) { ret some(ast.def_ty(defid)); }
        }
        case (ast.item_obj(_, _, _, ?odid, _)) {
            if (ns == ns_value) { ret some(ast.def_obj(odid.ctor)); }
            else { ret some(ast.def_obj(odid.ty)); }
        }
        case (_) { }
    }
    ret none[def];
}

fn lookup_in_mod_strict(&env e, def m, &span sp, ident id,
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

fn lookup_in_mod(&env e, def m, ident id, namespace ns, dir dr)
    -> Option.t[def] {
    auto defid = ast.def_id_of_def(m);
    if (defid._0 != ast.local_crate) { // Not in this crate
        auto cached = e.ext_cache.find(tup(defid,id));
        if (cached != none[def] && check_def_by_ns(Option.get(cached), ns)) {
            ret cached;
        }
        auto path = vec(id);
        if (defid._1 != -1) {
            path = e.ext_map.get(defid) + path;
        }
        auto fnd = lookup_external(e, defid._0, path, ns);
        if (fnd != none[def]) {
            e.ext_cache.insert(tup(defid,id), Option.get(fnd));
        }
        ret fnd;
    }
    alt (m) {
        case (ast.def_mod(?defid)) {
            ret lookup_in_regular_mod(e, defid, id, ns, dr);
        }
        case (ast.def_native_mod(?defid)) {
            ret lookup_in_native_mod(e, defid, id, ns);
        }
    }
}

fn found_view_item(&env e, @ast.view_item vi, namespace ns) -> Option.t[def] {
    alt (vi.node) {
        case (ast.view_item_use(_, _, _, ?cnum)) {
            ret some(ast.def_mod(tup(Option.get(cnum), -1)));
        }
        case (ast.view_item_import(_, _, ?defid)) {
            ret lookup_import(e, defid, ns);
        }
    }
}

fn lookup_in_regular_mod(&env e, def_id defid, ident id, namespace ns, dir dr)
    -> Option.t[def] {
    auto info = e.mod_map.get(defid._1);
    auto found = info.index.find(id);
    if (found == none[mod_index_entry] || 
        (dr == outside && !ast.is_exported(id, info.m))) {
        ret none[def];
    }
    alt (Option.get(found)) {
        case (mie_view_item(?view_item)) {
            ret found_view_item(e, view_item, ns);
        }
        case (mie_item(?item)) {
            ret found_def_item(item, ns);
        }
        case (mie_tag_variant(?item, ?variant_idx)) {
            alt (item.node) {
                case (ast.item_tag(_, ?variants, _, ?tid, _)) {
                    if (ns == ns_value) {
                        auto vid = variants.(variant_idx).node.id;
                        ret some(ast.def_variant(tid, vid));
                    } else {
                        ret none[def];
                    }
                }
            }
        }            
    }
}

fn lookup_in_native_mod(&env e, def_id defid, ident id, namespace ns)
    -> Option.t[def] {
    auto info = e.nmod_map.get(defid._1);
    auto found = info.index.find(id);
    if (found == none[native_mod_index_entry]) {
        ret none[def];
    }
    alt (Option.get(found)) {
        case (nmie_view_item(?view_item)) {
            ret found_view_item(e, view_item, ns);
        }
        case (nmie_item(?item)) {
            alt (item.node) {
                case (ast.native_item_ty(_, ?id)) {
                    if (ns == ns_type) {
                        ret some(ast.def_native_ty(id));
                    }
                }
                case (ast.native_item_fn(_, _, _, _, ?id, _)) {
                    if (ns == ns_value) {
                        ret some(ast.def_native_fn(id));
                    }
                }
            }
        }
        case (_) {}
    }
    ret none[def];
}

// Module indexing

fn index_mod(&ast._mod md) -> mod_index {
    auto index = new_str_hash[mod_index_entry]();

    for (@ast.view_item it in md.view_items) {
        alt (it.node) {
            case(ast.view_item_use(?id, _, _, _)) {
                index.insert(id, mie_view_item(it));
            }
            case(ast.view_item_import(?def_ident,_,_)) {
                index.insert(def_ident, mie_view_item(it));
            }
            case(ast.view_item_export(_)) {}
        }
    }

    for (@ast.item it in md.items) {
        alt (it.node) {
            case (ast.item_const(?id, _, _, _, _)) {
                index.insert(id, mie_item(it));
            }
            case (ast.item_fn(?id, _, _, _, _)) {
                index.insert(id, mie_item(it));
            }
            case (ast.item_mod(?id, _, _)) {
                index.insert(id, mie_item(it));
            }
            case (ast.item_native_mod(?id, _, _)) {
                index.insert(id, mie_item(it));
            }
            case (ast.item_ty(?id, _, _, _, _)) {
                index.insert(id, mie_item(it));
            }
            case (ast.item_tag(?id, ?variants, _, _, _)) {
                index.insert(id, mie_item(it));
                let uint variant_idx = 0u;
                for (ast.variant v in variants) {
                    index.insert(v.node.name,
                                 mie_tag_variant(it, variant_idx));
                    variant_idx += 1u;
                }
            }
            case (ast.item_obj(?id, _, _, _, _)) {
                index.insert(id, mie_item(it));
            }
        }
    }

    ret index;
}

fn index_nmod(&ast.native_mod md) -> nmod_index {
    auto index = new_str_hash[native_mod_index_entry]();

    for (@ast.view_item it in md.view_items) {
        alt (it.node) {
            case(ast.view_item_import(?def_ident,_,_)) {
                index.insert(def_ident, nmie_view_item(it));
            }
            case(ast.view_item_export(_)) {}
        }
    }

    for (@ast.native_item it in md.items) {
        alt (it.node) {
            case (ast.native_item_ty(?id, _)) {
                index.insert(id, nmie_item(it));
            }
            case (ast.native_item_fn(?id, _, _, _, _, _)) {
                index.insert(id, nmie_item(it));
            }
        }
    }

    ret index;
}

// External lookups

// FIXME creader should handle multiple namespaces
fn check_def_by_ns(def d, namespace ns) -> bool {
    ret alt (d) {
        case (ast.def_fn(?id)) { ns == ns_value }
        case (ast.def_obj(?id)) { ns == ns_value }
        case (ast.def_obj_field(?id)) { ns == ns_value }
        case (ast.def_mod(?id)) { true }
        case (ast.def_native_mod(?id)) { true }
        case (ast.def_const(?id)) { ns == ns_value }
        case (ast.def_arg(?id)) { ns == ns_value }
        case (ast.def_local(?id)) { ns == ns_value }
        case (ast.def_upvar(?id)) { ns == ns_value }
        case (ast.def_variant(_, ?id)) { ns == ns_value }
        case (ast.def_ty(?id)) { ns == ns_type }
        case (ast.def_binding(?id)) { ns == ns_type }
        case (ast.def_use(?id)) { true }
        case (ast.def_native_ty(?id)) { ns == ns_type }
        case (ast.def_native_fn(?id)) { ns == ns_value }
    };
}

fn lookup_external(&env e, int cnum, vec[ident] ids, namespace ns)
    -> Option.t[def] {
    auto found = creader.lookup_def(e.sess, cnum, ids);
    if (found != none[def]) {
        auto d = Option.get(found);
        if (!check_def_by_ns(d, ns)) { ret none[def]; }
        e.ext_map.insert(ast.def_id_of_def(d), ids);
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
