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


fn lookup_name_in_mod(session.session sess, &ast._mod m, &span sp,
                      ast.ident i) -> def_wrap {
    auto x = m.index.find(i);
    alt (x) {
        case (some[ast.mod_index_entry](?ent)) {
            alt (ent) {
                case (ast.mie_view_item(?view_item)) {
                    alt (view_item.node) {
                        case (ast.view_item_use(_, _, ?id)) {
                            ret def_wrap_use(view_item);
                        }
                        case (ast.view_item_import(?idents, ?the_def, ?d)) {
                            ret find_final_def(sess, m,
                                               view_item.span, idents);
                        }
                    }
                }
                case (ast.mie_item(?item)) {
                    alt (found_def_item(item)) {
                        case (some[def_wrap](?d)) {
                            ret d;
                        }
                        case (_) {
                            sess.span_err(sp, "unresolved name: " + i);
                        }
                    }
                }
            }
        }
        case (_) {
            sess.span_err(sp, "unresolved name: " + i);
        }
    }
    fail;
}

// Follow the path of an import and return what it ultimately points to.

fn find_final_def(session.session sess, &ast._mod m, &span sp,
                  vec[ident] idents)
    -> def_wrap {
    auto len = _vec.len[ident](idents);
    auto first = idents.(0);
    auto d = lookup_name_in_mod(sess, m, sp, first);
    if (len == 1u) {
        ret d;
    }
    alt (d) {
        case (def_wrap_mod(?i)) {
            auto new_idents = _vec.slice[ident](idents, 1u, len);
            alt (i.node) {
                case (ast.item_mod(_, ?m, _)) {
                    ret find_final_def(sess, m, sp, new_idents);
                }
            }
        }
        case (def_wrap_use(?c)) {
            sess.span_err(sp, "Crate access is not implemented");
        }
        case (_) {
            sess.span_err(sp, first + " is not a module or crate");
        }
    }
    fail;
}

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

fn lookup_name(&env e, ast.ident i) -> option.t[def] {

    fn found_decl_stmt(@ast.stmt s) -> option.t[def] {
        alt (s.node) {
            case (ast.stmt_decl(?d)) {
                alt (d.node) {
                    case (ast.decl_local(?loc)) {
                        ret some[def](ast.def_local(loc.id));
                    }
                    case (ast.decl_item(?it)) {
                        ret unwrap_def(found_def_item(it));
                    }
                }
            }
        }
        ret none[def];
    }

    fn found_def_view(@ast.view_item i) -> option.t[def] {
        alt (i.node) {
            case (ast.view_item_use(_, _, ?id)) {
                ret some[def](ast.def_use(id));
            }
            case (ast.view_item_import(?idents,?the_def, ?d)) {
                ret the_def;
            }
        }
        fail;
    }

    fn check_mod(ast.ident i, ast._mod m) -> option.t[def] {
        alt (m.index.find(i)) {
            case (some[ast.mod_index_entry](?ent)) {
                alt (ent) {
                    case (ast.mie_view_item(?view_item)) {
                        ret found_def_view(view_item);
                    }
                    case (ast.mie_item(?item)) {
                        ret unwrap_def(found_def_item(item));
                    }
                    case (ast.mie_tag_variant(?item, ?variant_idx)) {
                        alt (item.node) {
                            case (ast.item_tag(_, ?variants, _, ?tid)) {
                                auto vid = variants.(variant_idx).id;
                                auto t = ast.def_variant(tid, vid);
                                ret some[def](t);
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
        ret none[def];
    }


    fn in_scope(ast.ident i, &scope s) -> option.t[def] {
        alt (s) {

            case (scope_crate(?c)) {
                ret check_mod(i, c.node.module);
            }

            case (scope_item(?it)) {
                alt (it.node) {
                    case (ast.item_fn(_, ?f, ?ty_params, _, _)) {
                        for (ast.arg a in f.inputs) {
                            if (_str.eq(a.ident, i)) {
                                auto t = ast.def_arg(a.id);
                                ret some[def](t);
                            }
                        }
                        for (ast.ty_param tp in ty_params) {
                            if (_str.eq(tp.ident, i)) {
                                auto t = ast.def_ty_arg(tp.id);
                                ret some[def](t);
                            }
                        }
                    }
                    case (ast.item_obj(_, ?ob, ?ty_params, _, _)) {
                        for (ast.obj_field f in ob.fields) {
                            if (_str.eq(f.ident, i)) {
                                auto t = ast.def_obj_field(f.id);
                                ret some[def](t);
                            }
                        }
                        for (ast.ty_param tp in ty_params) {
                            if (_str.eq(tp.ident, i)) {
                                auto t = ast.def_ty_arg(tp.id);
                                ret some[def](t);
                            }
                        }
                    }
                    case (ast.item_mod(_, ?m, _)) {
                        ret check_mod(i, m);
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
                        ret some[def](t);
                    }
                    case (_) { /* fall through */  }
                }
            }
        }
        ret none[def];
    }

    ret std.list.find[scope,def](e.scopes, bind in_scope(i, _));
}

fn fold_pat_tag(&env e, &span sp, ident i,
                vec[@ast.pat] args, option.t[ast.variant_def] old_def,
                ann a) -> @ast.pat {
    auto new_def;
    alt (lookup_name(e, i)) {
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

fn fold_expr_name(&env e, &span sp,
                  &ast.name n, &option.t[def] d, ann a) -> @ast.expr {

    if (_vec.len[@ast.ty](n.node.types) > 0u) {
        e.sess.unimpl("resolving name expr with ty params");
    }

    auto d_ = lookup_name(e, n.node.ident);

    alt (d_) {
        case (some[def](_)) {
            // log "resolved name " + n.node.ident;
        }
        case (none[def]) {
            e.sess.span_err(sp, "unresolved name: " + n.node.ident);
        }
    }

    ret @fold.respan[ast.expr_](sp, ast.expr_name(n, d_, a));
}

fn global_mod(&env e) -> ast._mod {
    alt (e.scopes) {
        case (cons[scope](?hd, _)) {
            alt(hd) {
                case (scope_crate(?c)) {
                    ret c.node.module;
                }
            }
        }
    }
    fail;
}

fn fold_view_item_import(&env e, &span sp,
                         vec[ident] is,
                         option.t[def] the_def,
                         ast.def_id id) -> @ast.view_item {
    // Produce errors for invalid imports
    auto len = _vec.len[ast.ident](is);
    auto last_id = is.(len - 1u);
    auto g_mod = global_mod(e);
    auto d = find_final_def(e.sess, g_mod, sp, is);
    auto x = unwrap_def(some(d));
    ret @fold.respan[ast.view_item_](sp,
                                     ast.view_item_import(is, x, id));
}


fn fold_ty_path(&env e, &span sp, ast.path p,
                &option.t[def] d) -> @ast.ty {

    let uint len = _vec.len[ast.name](p);
    check (len != 0u);
    if (len > 1u) {
        e.sess.unimpl("resolving path ty with >1 component");
    }

    let ast.name n = p.(0);

    if (_vec.len[@ast.ty](n.node.types) > 0u) {
        e.sess.unimpl("resolving path ty with ty params");
    }

    auto d_ = lookup_name(e, n.node.ident);

    alt (d_) {
        case (some[def](_)) {
            // log "resolved name " + n.node.ident;
        }
        case (none[def]) {
            e.sess.span_err(sp, "unresolved name: " + n.node.ident);
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

fn resolve_imports(session.session sess, @ast.crate crate)
    -> @ast.crate {
   let fold.ast_fold[env] fld = fold.new_identity_fold[env]();
   fld = @rec( fold_view_item_import = bind fold_view_item_import(_,_,_,_,_),
                    update_env_for_crate = bind update_env_for_crate(_,_)
                    with *fld );

    auto e = rec(scopes = nil[scope],
                 sess = sess);

    ret fold.fold_crate[env](e, fld, crate);
}

fn resolve_crate(session.session sess, @ast.crate crate) -> @ast.crate {

    let fold.ast_fold[env] fld = fold.new_identity_fold[env]();

    auto new_crate = resolve_imports(sess, crate);

    fld = @rec( fold_pat_tag = bind fold_pat_tag(_,_,_,_,_,_),
                fold_expr_name = bind fold_expr_name(_,_,_,_,_),
                fold_ty_path = bind fold_ty_path(_,_,_,_),
                update_env_for_crate = bind update_env_for_crate(_,_),
                update_env_for_item = bind update_env_for_item(_,_),
                update_env_for_block = bind update_env_for_block(_,_),
                update_env_for_arm = bind update_env_for_arm(_,_)
                with *fld );

    auto e = rec(scopes = nil[scope],
                 sess = sess);

    ret fold.fold_crate[env](e, fld, new_crate);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
