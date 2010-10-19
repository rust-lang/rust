import front.ast;
import front.ast.ident;
import front.ast.def;
import driver.session;
import util.common.span;
import std.map.hashmap;
import std.list.list;
import std.list.nil;
import std.list.cons;
import std.util.option;
import std.util.some;
import std.util.none;
import std._str;

tag scope {
    scope_crate(@ast.crate);
    scope_item(@ast.item);
    scope_block(ast.block);
}

type env = rec(list[scope] scopes,
               session.session sess);

fn lookup_name(&env e, ast.ident i) -> option[def] {

    log "resolving name " + i;

    fn found_def_item(@ast.item i) -> option[def] {
        alt (i.node) {
            case (ast.item_fn(_, _, ?id)) {
                ret some[def](ast.def_fn(id));
            }
            case (ast.item_mod(_, _, ?id)) {
                ret some[def](ast.def_mod(id));
            }
            case (ast.item_ty(_, _, ?id)) {
                ret some[def](ast.def_ty(id));
            }
        }
    }

    fn found_decl_stmt(@ast.stmt s) -> option[def] {
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

    fn check_mod(ast.ident i, ast._mod m) -> option[def] {
        alt (m.index.find(i)) {
            case (some[uint](?ix)) {
                ret found_def_item(m.items.(ix));
            }
        }
        ret none[def];
    }


    fn in_scope(ast.ident i, &scope s) -> option[def] {
        alt (s) {

            case (scope_crate(?c)) {
                ret check_mod(i, c.node.module);
            }

            case (scope_item(?it)) {
                alt (it.node) {
                    case (ast.item_fn(_, ?f, _)) {
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
                  &option[def] d, option[@ast.ty] t) -> @ast.expr {

    auto d_ = lookup_name(e, n.node.ident);

    alt (d_) {
        case (some[def](_)) {
            log "resolved name " + n.node.ident;
        }
        case (none[def]) {
            e.sess.err("unresolved name: " + n.node.ident);
        }
    }

    ret @fold.respan[ast.expr_](sp, ast.expr_name(n, d_, t));
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
