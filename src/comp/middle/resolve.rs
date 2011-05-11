import front.ast;
import front.ast.ident;
import front.ast.def;
import front.ast.def_id;
import front.ast.ann;
import front.creader;
import driver.session.session;
import util.common.new_def_hash;
import util.common.new_int_hash;
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

tag scope {
    scope_crate(@ast.crate);
    scope_item(@ast.item);
    scope_native_item(@ast.native_item);
    scope_loop(@ast.decl); // there's only 1 decl per loop.
    scope_block(ast.block);
    scope_arm(ast.arm);
}

tag wrap_mod {
    wmod(ast._mod);
    wnmod(ast.native_mod);
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

type env = rec(hashmap[ast.def_num,import_state] imports,
               hashmap[ast.def_num,@wrap_mod] mod_map,
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

fn resolve_crate(session sess, @ast.crate crate) -> @ast.crate {
    auto e = @rec(imports = new_int_hash[import_state](),
                  mod_map = new_int_hash[@wrap_mod](),
                  ext_map = new_def_hash[vec[ident]](),
                  ext_cache = new_ext_hash(),
                  sess = sess);
    map_crate(e, *crate);
    resolve_imports(*e);
    ret resolve_names(e, *crate);
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
                e.mod_map.insert(defid._1, @wmod(md));
            }
            case (ast.item_native_mod(_, ?nmd, ?defid)) {
                e.mod_map.insert(defid._1, @wnmod(nmd));
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

fn resolve_names(&@env e, &ast.crate c) -> @ast.crate {
    auto fld = @rec(fold_pat_tag = bind fold_pat_tag(e,_,_,_,_,_,_),
                    fold_expr_path = bind fold_expr_path(e,_,_,_,_,_),
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

// We received a path expression of the following form:
//
//     a.b.c.d
//
// Somewhere along this path there might be a split from a path-expr
// to a runtime field-expr. For example:
//
//     'a' could be the name of a variable in the local scope
//     and 'b.c.d' could be a field-sequence inside it.
//
// Or:
//
//     'a.b' could be a module path to a constant record, and 'c.d'
//     could be a field within it.
//
// Our job here is to figure out what the prefix of 'a.b.c.d' is that
// corresponds to a static binding-name (a module or slot, with no type info)
// and split that off as the 'primary' expr_path, with secondary expr_field
// expressions tacked on the end.

fn fold_expr_path(@env e, &list[scope] sc, &span sp, &ast.path p,
                  &Option.t[def] d, &ann a) -> @ast.expr {
    auto idents = p.node.idents;
    auto n_idents = Vec.len(idents);
    assert (n_idents != 0u);

    auto dcur = lookup_in_scope_strict(*e, sc, sp, idents.(0), ns_value);
    auto i = 1u;
    while (i < n_idents) {
        if (!is_module(dcur)) { break; }
        dcur = lookup_in_mod_strict(*e, dcur, sp, idents.(i), ns_value,
                                    outside);
        i += 1u;
    }
    if (is_module(dcur)) {
        e.sess.span_err(sp, "can't refer to a module as a first-class value");
    }

    p = rec(node=rec(idents=Vec.slice(idents, 0u, i) with p.node) with p);
    auto ex = @fold.respan(sp, ast.expr_path(p, some(dcur), a));
    while (i < n_idents) {
        ex = @fold.respan(sp, ast.expr_field(ex, idents.(i), a));
        i += 1u;
    }
    ret ex;
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
        e.sess.span_err(sp, Str.connect(idents, ".") +
                        " is a module, not a " + ns_name(ns));
    }
    ret dcur;
}
                      
fn fold_pat_tag(@env e, &list[scope] sc, &span sp, &ast.path p,
                &vec[@ast.pat] args, &Option.t[ast.variant_def] old_def,
                &ann a) -> @ast.pat {
    alt (lookup_path_strict(*e, sc, sp, p.node.idents, ns_value)) {
        case (ast.def_variant(?did, ?vid)) {
            auto new_def = some[ast.variant_def](tup(did, vid));
            ret @fold.respan[ast.pat_](sp, ast.pat_tag(p, args, new_def, a));
        }
        case (_) {
            e.sess.span_err(sp, "not a tag variant: " +
                            Str.connect(p.node.idents, "."));
            fail;
        }
    }
}

fn fold_ty_path(@env e, &list[scope] sc, &span sp, &ast.path p,
                &Option.t[def] d) -> @ast.ty {
    auto new_def = lookup_path_strict(*e, sc, sp, p.node.idents, ns_type);
    ret @fold.respan[ast.ty_](sp, ast.ty_path(p, some(new_def)));
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
                ret lookup_in_regular_mod(e, c.node.module, id, ns, inside);
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
                    case (ast.item_mod(_, ?m, _)) {
                        ret lookup_in_regular_mod(e, m, id, ns, inside);
                    }
                    case (ast.item_native_mod(_, ?m, _)) {
                        ret lookup_in_native_mod(e, m, id, ns);
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
                    alt (a.index.find(id)) {
                        case (some[def_id](?did)) {
                            ret some(ast.def_binding(did));
                        }
                        case (_) {}
                    }
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
        // def_num=-1 is a kludge to overload def_mod for external crates,
        // since those don't get a def num
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
            alt (*e.mod_map.get(defid._1)) {
                case (wmod(?m)) {
                    ret lookup_in_regular_mod(e, m, id, ns, dr);
                }
            }
        }
        case (ast.def_native_mod(?defid)) {
            alt (*e.mod_map.get(defid._1)) {
                case (wnmod(?m)) {
                    ret lookup_in_native_mod(e, m, id, ns);
                }
            }
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

fn lookup_in_regular_mod(&env e, &ast._mod md, ident id, namespace ns, dir dr)
    -> Option.t[def] {
    auto found = md.index.find(id);
    if (found == none[ast.mod_index_entry] || 
        (dr == outside && !ast.is_exported(id, md))) {
        ret none[def];
    }
    alt (Option.get(found)) {
        case (ast.mie_view_item(?view_item)) {
            ret found_view_item(e, view_item, ns);
        }
        case (ast.mie_item(?item)) {
            ret found_def_item(item, ns);
        }
        case (ast.mie_tag_variant(?item, ?variant_idx)) {
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

fn lookup_in_native_mod(&env e, &ast.native_mod md, ident id, namespace ns)
    -> Option.t[def] {
    auto found = md.index.find(id);
    if (found == none[ast.native_mod_index_entry]) {
        ret none[def];
    }
    alt (Option.get(found)) {
        case (ast.nmie_view_item(?view_item)) {
            ret found_view_item(e, view_item, ns);
        }
        case (ast.nmie_item(?item)) {
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
