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

type env = list[scope];

fn lookup_name(&env e, ast.ident i) -> option[def] {

    log "resolving name " + i;

    fn check_mod(ast.ident i, ast._mod m) -> option[def] {
        alt (m.find(i)) {
            case (some[@ast.item](?it)) {
                alt (it.node) {
                    case (ast.item_fn(_, ?id)) {
                        ret some[def](ast.def_fn(id));
                    }
                    case (ast.item_mod(_, ?id)) {
                        ret some[def](ast.def_mod(id));
                    }
                    case (ast.item_ty(_, ?id)) {
                        ret some[def](ast.def_ty(id));
                    }
                }
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
                    case (ast.item_fn(?f, _)) {
                        for (ast.arg a in f.inputs) {
                            if (_str.eq(a.ident, i)) {
                                ret some[def](ast.def_arg(a.id));
                            }
                        }
                    }
                    case (ast.item_mod(?m, _)) {
                        ret check_mod(i, m);
                    }
                }
            }
        }
        ret none[def];
    }

    ret std.list.find[scope,def](e, bind in_scope(i, _));
}

fn fold_expr_name(&env e, &span sp, &ast.name n,
                  &option[def] d) -> @ast.expr {

    auto d_ = lookup_name(e, n.node.ident);

    alt (d_) {
        case (some[def](_)) {
            log "resolved name " + n.node.ident;
        }
        case (none[def]) {
            log "unresolved name " + n.node.ident;
        }
    }

    ret @fold.respan[ast.expr_](sp, ast.expr_name(n, d_));
}

fn update_env_for_crate(&env e, @ast.crate c) -> env {
    ret cons[scope](scope_crate(c), @e);
}

fn update_env_for_item(&env e, @ast.item i) -> env {
    ret cons[scope](scope_item(i), @e);
}

fn update_env_for_block(&env e, ast.block b) -> env {
    ret cons[scope](scope_block(b), @e);
}

fn resolve_crate(session.session sess, @ast.crate crate) -> @ast.crate {
    let fold.ast_fold[env] fld = fold.new_identity_fold[env]();
    fld = @rec( fold_expr_name = bind fold_expr_name(_,_,_,_),
                update_env_for_crate = bind update_env_for_crate(_,_),
                update_env_for_item = bind update_env_for_item(_,_),
                update_env_for_block = bind update_env_for_block(_,_)
                with *fld );
    ret fold.fold_crate[env](nil[scope], fld, crate);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
