import front.ast;
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

fn resolve_name(&env e, &span sp, ast.name_ n) -> ast.name {

    log "resolving name " + n.ident;

    fn in_scope(ast.ident i, &scope s) -> option[scope] {
        alt (s) {
            case (scope_crate(?c)) {
                if (c.node.module.contains_key(i)) {
                    ret some[scope](s);
                }
            }
            case (scope_item(?it)) {
                alt (it.node) {
                    case (ast.item_fn(?f, _)) {
                        for (ast.input inp in f.inputs) {
                            if (_str.eq(inp.ident, i)) {
                                ret some[scope](s);
                            }
                        }
                    }
                    case (ast.item_mod(?m)) {
                        if (m.contains_key(i)) {
                            ret some[scope](s);
                        }
                    }
                }
            }
        }
        ret none[scope];
    }

    alt (std.list.find[scope](e, bind in_scope(n.ident, _))) {
        case (some[scope](?s)) {
            log "resolved name " + n.ident;
        }
        case (none[scope]) {
            log "unresolved name " + n.ident;
        }
    }

    ret fold.respan[ast.name_](sp, n);
}

fn update_env_for_crate(&env e, @ast.crate c) -> env {
    log "updating env with crate";
    ret cons[scope](scope_crate(c), @e);
}

fn update_env_for_item(&env e, @ast.item i) -> env {
    log "updating env with item";
    ret cons[scope](scope_item(i), @e);
}

fn update_env_for_block(&env e, ast.block b) -> env {
    log "updating env with block";
    ret cons[scope](scope_block(b), @e);
}

fn resolve_crate(session.session sess, @ast.crate crate) -> @ast.crate {
    let fold.ast_fold[env] fld = fold.new_identity_fold[env]();
    fld = @rec( fold_name = bind resolve_name(_,_,_),
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
