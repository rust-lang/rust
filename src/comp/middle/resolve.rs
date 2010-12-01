import front.ast;
import front.ast.ident;
import front.ast.def;
import front.ast.ann;
import driver.session;
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
}

type env = rec(list[scope] scopes,
               session.session sess);

fn lookup_name(&env e, ast.ident i) -> option.t[def] {

    // log "resolving name " + i;

    fn found_def_item(@ast.item i) -> option.t[def] {
        alt (i.node) {
            case (ast.item_fn(_, _, _, ?id, _)) {
                ret some[def](ast.def_fn(id));
            }
            case (ast.item_mod(_, _, ?id)) {
                ret some[def](ast.def_mod(id));
            }
            case (ast.item_ty(_, _, _, ?id, _)) {
                ret some[def](ast.def_ty(id));
            }
            case (ast.item_tag(_, _, _, ?id)) {
                ret some[def](ast.def_ty(id));
            }
        }
    }

    fn found_decl_stmt(@ast.stmt s) -> option.t[def] {
        alt (s.node) {
            case (ast.stmt_decl(?d)) {
                alt (d.node) {
                    case (ast.decl_local(?loc)) {
                        ret some[def](ast.def_local(loc.id));
                    }
                    case (ast.decl_item(?it)) {
                        ret found_def_item(it);
                    }
                }
            }
        }
        ret none[def];
    }

    fn check_mod(ast.ident i, ast._mod m) -> option.t[def] {
        alt (m.index.find(i)) {
            case (some[ast.mod_index_entry](?ent)) {
                alt (ent) {
                    case (ast.mie_item(?ix)) {
                        ret found_def_item(m.items.(ix));
                    }
                    case (ast.mie_tag_variant(?item_idx, ?variant_idx)) {
                        alt (m.items.(item_idx).node) {
                            case (ast.item_tag(_, ?variants, _, ?tid)) {
                                auto vid = variants.(variant_idx).id;
                                ret some[def](ast.def_variant(tid, vid));
                            }
                            case (_) {
                                log "tag item not actually a tag";
                                fail;
                            }
                        }
                    }
                }
            }
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
                    case (ast.item_fn(_, ?f, _, _, _)) {
                        for (ast.arg a in f.inputs) {
                            if (_str.eq(a.ident, i)) {
                                ret some[def](ast.def_arg(a.id));
                            }
                        }
                    }
                    case (ast.item_mod(_, ?m, _)) {
                        ret check_mod(i, m);
                    }
                }
            }

            case (scope_block(?b)) {
                alt (b.node.index.find(i)) {
                    case (some[uint](?ix)) {
                        ret found_decl_stmt(b.node.stmts.(ix));
                    }
                }
            }
        }
        ret none[def];
    }

    ret std.list.find[scope,def](e.scopes, bind in_scope(i, _));
}

fn fold_expr_name(&env e, &span sp, &ast.name n,
                  &option.t[def] d, ann a) -> @ast.expr {

    if (_vec.len[@ast.ty](n.node.types) > 0u) {
        e.sess.unimpl("resoling name expr with ty params");
    }

    auto d_ = lookup_name(e, n.node.ident);

    alt (d_) {
        case (some[def](_)) {
            // log "resolved name " + n.node.ident;
        }
        case (none[def]) {
            e.sess.err("unresolved name: " + n.node.ident);
        }
    }

    ret @fold.respan[ast.expr_](sp, ast.expr_name(n, d_, a));
}

fn fold_ty_path(&env e, &span sp, ast.path p,
                &option.t[def] d) -> @ast.ty {

    let uint len = _vec.len[ast.name](p);
    check (len != 0u);
    if (len > 1u) {
        e.sess.unimpl("resoling path ty with >1 component");
    }

    let ast.name n = p.(0);

    if (_vec.len[@ast.ty](n.node.types) > 0u) {
        e.sess.unimpl("resoling path ty with ty params");
    }

    auto d_ = lookup_name(e, n.node.ident);

    alt (d_) {
        case (some[def](_)) {
            // log "resolved name " + n.node.ident;
        }
        case (none[def]) {
            e.sess.err("unresolved name: " + n.node.ident);
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

fn resolve_crate(session.session sess, @ast.crate crate) -> @ast.crate {

    let fold.ast_fold[env] fld = fold.new_identity_fold[env]();

    fld = @rec( fold_expr_name = bind fold_expr_name(_,_,_,_,_),
                fold_ty_path = bind fold_ty_path(_,_,_,_),
                update_env_for_crate = bind update_env_for_crate(_,_),
                update_env_for_item = bind update_env_for_item(_,_),
                update_env_for_block = bind update_env_for_block(_,_)
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
