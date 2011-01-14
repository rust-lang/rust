import front.ast;
import front.ast.ident;
import front.ast.def;
import front.ast.ann;
import driver.session;
import util.common.new_def_hash;
import util.common.span;
import std.map.hashmap;
import std.list.list;
import std.list.nil;
import std.list.cons;
import std.option;
import std.option.some;
import std.option.none;
import std._str;
import std._vec;

tag scope {
    scope_crate(@ast.crate);
    scope_item(@ast.item);
    scope_block(ast.block);
    scope_arm(ast.arm);
}

type env = rec(list[scope] scopes,
               session.session sess);

type import_map = std.map.hashmap[ast.def_id,def_wrap];

// A simple wrapper over defs that stores a bit more information about modules
// and uses so that we can use the regular lookup_name when resolving imports.
tag def_wrap {
    def_wrap_use(@ast.view_item);
    def_wrap_mod(@ast.item);
    def_wrap_other(def);
}

fn unwrap_def(option.t[def_wrap] d_) -> option.t[def] {
    alt (d_) {
        case (none[def_wrap]) {
            ret none[def];
        }
        case (some[def_wrap](?d)) {
            alt (d) {
                case (def_wrap_use(?it)) {
                    alt (it.node) {
                        case (ast.view_item_use(_, _, ?id)) {
                            ret some[def](ast.def_use(id));
                        }
                    }
                    fail;
                }
                case (def_wrap_mod(?i)) {
                    alt (i.node) {
                        case (ast.item_mod(_, _, ?id)) {
                            ret some[def](ast.def_mod(id));
                        }
                    }
                    fail;
                }
                case (def_wrap_other(?d)) {
                    ret some[def](d);
                }
            }
        }
    }
    fail;
}

// Follow the path of an import and return what it ultimately points to.

fn find_final_def(&env e, &span sp, vec[ident] idents) -> option.t[def_wrap] {
    auto len = _vec.len[ident](idents);
    auto first = idents.(0);
    auto d_ = lookup_name(e, none[import_map], first);
    if (len == 1u) {
        ret d_;
    }
    alt (d_) {
        case (none[def_wrap]) {
            e.sess.span_err(sp, "unresolved name: " + first);
            ret d_;
        }
        case (some[def_wrap](?d)) {
            alt (d) {
                case (def_wrap_mod(?i)) {
                    auto new_idents = _vec.slice[ident](idents, 1u, len);
                    auto tmp_e = rec(scopes = nil[scope],
                                     sess = e.sess);
                    auto new_e = update_env_for_item(tmp_e, i);
                    ret find_final_def(new_e, sp, new_idents);
                }
                case (def_wrap_use(?c)) {
                    e.sess.span_err(sp, "Crate access is not implemented");
                }
                case (_) {
                    e.sess.span_err(sp, first + " is not a module or crate");
                }
            }
        }
    }
    fail;
}

fn lookup_name(&env e, option.t[import_map] index,
               ast.ident i) -> option.t[def_wrap] {

    // log "resolving name " + i;

    fn found_def_item(@ast.item i) -> option.t[def_wrap] {
        alt (i.node) {
            case (ast.item_const(_, _, _, ?id, _)) {
                ret some[def_wrap](def_wrap_other(ast.def_const(id)));
            }
            case (ast.item_fn(_, _, _, ?id, _)) {
                ret some[def_wrap](def_wrap_other(ast.def_fn(id)));
            }
            case (ast.item_mod(_, _, ?id)) {
                ret some[def_wrap](def_wrap_mod(i));
            }
            case (ast.item_ty(_, _, _, ?id, _)) {
                ret some[def_wrap](def_wrap_other(ast.def_ty(id)));
            }
            case (ast.item_tag(_, _, _, ?id)) {
                ret some[def_wrap](def_wrap_other(ast.def_ty(id)));
            }
            case (ast.item_obj(_, _, _, ?id, _)) {
                ret some[def_wrap](def_wrap_other(ast.def_obj(id)));
            }
        }
    }

    fn found_decl_stmt(@ast.stmt s) -> option.t[def_wrap] {
        alt (s.node) {
            case (ast.stmt_decl(?d)) {
                alt (d.node) {
                    case (ast.decl_local(?loc)) {
                        auto t = ast.def_local(loc.id);
                        ret some[def_wrap](def_wrap_other(t));
                    }
                    case (ast.decl_item(?it)) {
                        ret found_def_item(it);
                    }
                }
            }
        }
        ret none[def_wrap];
    }

    fn found_def_view(&env e, option.t[import_map] index,
                      @ast.view_item i) -> option.t[def_wrap] {
        alt (i.node) {
            case (ast.view_item_use(_, _, ?id)) {
                ret some[def_wrap](def_wrap_use(i));
            }
            case (ast.view_item_import(?idents,?d)) {
                alt (index) {
                    case (some[import_map](?idx)) {
                        ret idx.find(d);
                    }
                    case (none[import_map]) {
                        ret find_final_def(e, i.span, idents);
                    }
                }
            }
        }
        fail;
    }

    fn check_mod(&env e, option.t[import_map] index, ast.ident i,
                 ast._mod m) -> option.t[def_wrap] {
        alt (m.index.find(i)) {
            case (some[ast.mod_index_entry](?ent)) {
                alt (ent) {
                    case (ast.mie_view_item(?view_item)) {
                        ret found_def_view(e, index, view_item);
                    }
                    case (ast.mie_item(?item)) {
                        ret found_def_item(item);
                    }
                    case (ast.mie_tag_variant(?item, ?variant_idx)) {
                        alt (item.node) {
                            case (ast.item_tag(_, ?variants, _, ?tid)) {
                                auto vid = variants.(variant_idx).id;
                                auto t = ast.def_variant(tid, vid);
                                ret some[def_wrap](def_wrap_other(t));
                            }
                            case (_) {
                                log "tag item not actually a tag";
                                fail;
                            }
                        }
                    }
                }
            }
            case (none[ast.mod_index_entry]) { /* fall through */ }
        }
        ret none[def_wrap];
    }


    fn in_scope(ast.ident i, env e, option.t[import_map] index,
                &scope s) -> option.t[def_wrap] {
        alt (s) {

            case (scope_crate(?c)) {
                ret check_mod(e, index, i, c.node.module);
            }

            case (scope_item(?it)) {
                alt (it.node) {
                    case (ast.item_fn(_, ?f, ?ty_params, _, _)) {
                        for (ast.arg a in f.inputs) {
                            if (_str.eq(a.ident, i)) {
                                auto t = ast.def_arg(a.id);
                                ret some[def_wrap](def_wrap_other(t));
                            }
                        }
                        for (ast.ty_param tp in ty_params) {
                            if (_str.eq(tp.ident, i)) {
                                auto t = ast.def_ty_arg(tp.id);
                                ret some[def_wrap](def_wrap_other(t));
                            }
                        }
                    }
                    case (ast.item_obj(_, ?ob, ?ty_params, _, _)) {
                        for (ast.obj_field f in ob.fields) {
                            if (_str.eq(f.ident, i)) {
                                auto t = ast.def_obj_field(f.id);
                                ret some[def_wrap](def_wrap_other(t));
                            }
                        }
                        for (ast.ty_param tp in ty_params) {
                            if (_str.eq(tp.ident, i)) {
                                auto t = ast.def_ty_arg(tp.id);
                                ret some[def_wrap](def_wrap_other(t));
                            }
                        }
                    }
                    case (ast.item_mod(_, ?m, _)) {
                        ret check_mod(e, index, i, m);
                    }
                    case (_) { /* fall through */ }
                }
            }

            case (scope_block(?b)) {
                alt (b.node.index.find(i)) {
                    case (some[uint](?ix)) {
                        ret found_decl_stmt(b.node.stmts.(ix));
                    }
                    case (_) { /* fall through */  }
                }
            }

            case (scope_arm(?a)) {
                alt (a.index.find(i)) {
                    case (some[ast.def_id](?did)) {
                        auto t = ast.def_binding(did);
                        ret some[def_wrap](def_wrap_other(t));
                    }
                    case (_) { /* fall through */  }
                }
            }
        }
        ret none[def_wrap];
    }

    ret std.list.find[scope,def_wrap](e.scopes,
                                      bind in_scope(i, e, index, _));
}

fn fold_pat_tag(&env e, &span sp, import_map index, ident i,
                vec[@ast.pat] args, option.t[ast.variant_def] old_def,
                ann a) -> @ast.pat {
    auto new_def;
    alt (unwrap_def(lookup_name(e, some(index), i))) {
        case (some[def](?d)) {
            alt (d) {
                case (ast.def_variant(?did, ?vid)) {
                    new_def = some[ast.variant_def](tup(did, vid));
                }
                case (_) {
                    e.sess.span_err(sp, "not a tag variant: " + i);
                    new_def = none[ast.variant_def];
                }
            }
        }
        case (none[def]) {
            new_def = none[ast.variant_def];
            e.sess.span_err(sp, "unresolved name: " + i);
        }
    }

    ret @fold.respan[ast.pat_](sp, ast.pat_tag(i, args, new_def, a));
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

fn fold_expr_path(&env e, &span sp, import_map index,
                  &ast.path p, &option.t[def] d, ann a) -> @ast.expr {

    if (_vec.len[@ast.ty](p.node.types) > 0u) {
        e.sess.unimpl("resolving name expr with ty params");
    }

    auto n_idents = _vec.len[ast.ident](p.node.idents);

    check (n_idents != 0u);
    auto id0 = p.node.idents.(0);

    auto d_ = unwrap_def(lookup_name(e, some(index), id0));

    alt (d_) {
        case (some[def](_)) {
            // log "resolved name " + n.node.ident;
        }
        case (none[def]) {
            e.sess.span_err(sp, "unresolved name: " + id0);
        }
    }

    // FIXME: once espindola's modifications to lookup land, actually step
    // through the path doing speculative lookup, and extend the maximal
    // static prefix. For now we are always using the minimal prefix: first
    // ident is static anchor, rest turn into fields.

    auto p_ = rec(node=rec(idents = vec(id0) with p.node) with p);
    auto ex = @fold.respan[ast.expr_](sp, ast.expr_path(p_, d_, a));
    auto i = 1u;
    while (i < n_idents) {
        auto id = p.node.idents.(i);
        ex = @fold.respan[ast.expr_](sp, ast.expr_field(ex, id, a));
        i += 1u;
    }
    ret ex;
}

fn fold_view_item_import(&env e, &span sp,
                         import_map index,
                         vec[ident] is, ast.def_id id) -> @ast.view_item {
    // Produce errors for invalid imports
    auto len = _vec.len[ast.ident](is);
    auto last_id = is.(len - 1u);
    auto d = find_final_def(e, sp, is);
    alt (d) {
        case (none[def_wrap]) {
            e.sess.span_err(sp, "unresolved name: " + last_id);
        }
        case (some[def_wrap](?d2)) {
            index.insert(id, d2);
        }
    }

    ret @fold.respan[ast.view_item_](sp, ast.view_item_import(is, id));
}


fn fold_ty_path(&env e, &span sp, import_map index, ast.path p,
                &option.t[def] d) -> @ast.ty {

    let uint len = _vec.len[ast.ident](p.node.idents);
    check (len != 0u);
    if (len > 1u) {
        e.sess.unimpl("resolving path ty with >1 component");
    }

    if (_vec.len[@ast.ty](p.node.types) > 0u) {
        e.sess.unimpl("resolving path ty with ty params");
    }

    auto d_ = unwrap_def(lookup_name(e, some(index), p.node.idents.(0)));

    alt (d_) {
        case (some[def](?d)) {
            // log "resolved name " + n.node.ident;
        }
        case (none[def]) {
            e.sess.span_err(sp, "unresolved name: " + p.node.idents.(0));
        }
    }

    ret @fold.respan[ast.ty_](sp, ast.ty_path(p, d_));
}

fn update_env_for_crate(&env e, @ast.crate c) -> env {
    ret rec(scopes = cons[scope](scope_crate(c), @e.scopes) with e);
}

fn update_env_for_item(&env e, @ast.item i) -> env {
    ret rec(scopes = cons[scope](scope_item(i), @e.scopes) with e);
}

fn update_env_for_block(&env e, &ast.block b) -> env {
    ret rec(scopes = cons[scope](scope_block(b), @e.scopes) with e);
}

fn update_env_for_arm(&env e, &ast.arm p) -> env {
    ret rec(scopes = cons[scope](scope_arm(p), @e.scopes) with e);
}

fn resolve_crate(session.session sess, @ast.crate crate) -> @ast.crate {

    let fold.ast_fold[env] fld = fold.new_identity_fold[env]();

    auto import_index = new_def_hash[def_wrap]();
    fld = @rec( fold_pat_tag = bind fold_pat_tag(_,_,import_index,_,_,_,_),
                fold_expr_path = bind fold_expr_path(_,_,import_index,_,_,_),
                fold_view_item_import
                    = bind fold_view_item_import(_,_,import_index,_,_),
                fold_ty_path = bind fold_ty_path(_,_,import_index,_,_),
                update_env_for_crate = bind update_env_for_crate(_,_),
                update_env_for_item = bind update_env_for_item(_,_),
                update_env_for_block = bind update_env_for_block(_,_),
                update_env_for_arm = bind update_env_for_arm(_,_)
                with *fld );

    auto e = rec(scopes = nil[scope],
                 sess = sess);

    ret fold.fold_crate[env](e, fld, crate);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
