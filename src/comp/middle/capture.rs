import driver.session;
import front.ast;
import std.map.hashmap;
import std.option;
import std.option.some;
import std.option.none;
import std._int;
import util.common;

type fn_id_of_local = std.map.hashmap[ast.def_id, ast.def_id];
type env = rec(option.t[ast.def_id] current_context, // fn or obj
               fn_id_of_local idmap,
               session.session sess);

fn update_env_for_item(&env e, @ast.item i) -> env {
    alt (i.node) {
        case (ast.item_fn(?name, _, _, ?id, _)) {
            ret rec(current_context = some(id) with e);
        }
        case (ast.item_obj(_, _, _, ?ids, _)) {
            ret rec(current_context = some(ids.ty) with e);
        }
        case (_) {
            ret e;
        }
    }
}

fn update_env_for_expr(&env e, @ast.expr x) -> env {
    alt (x.node) {
        case (ast.expr_for(?d, _, _, _)) {
            alt (d.node) {
                case (ast.decl_local(?local)) {
                    auto curr_context =
                        option.get[ast.def_id](e.current_context);
                    e.idmap.insert(local.id, curr_context);
                }
                case (_) {
                }
            }
        }
        case (ast.expr_for_each(?d, _, _, _)) {
            alt (d.node) {
                case (ast.decl_local(?local)) {
                    auto curr_context =
                        option.get[ast.def_id](e.current_context);
                    e.idmap.insert(local.id, curr_context);
                }
                case (_) {
                }
            }
        }
        case (_) { }
    }
    ret e;
}

fn update_env_for_block(&env e, &ast.block b) -> env {
    auto curr_context = option.get[ast.def_id](e.current_context);

    for each (@tup(ast.ident, ast.block_index_entry) it in
              b.node.index.items()) {
        alt (it._1) {
            case (ast.bie_local(?local)) {
                e.idmap.insert(local.id, curr_context);
            }
            case (_) {
            }
        }
    }

    ret e;
}

fn fold_expr_path(&env e, &ast.span sp, &ast.path p, &option.t[ast.def] d,
                  ast.ann a) -> @ast.expr {
    auto local_id;
    alt (option.get[ast.def](d)) {
        case (ast.def_local(?id)) {
            local_id = id;
        }
        case (_) {
            ret @fold.respan[ast.expr_](sp, ast.expr_path(p, d, a));
        }
    }

    auto curr_context = option.get[ast.def_id](e.current_context);
    auto x = ast.def_id_of_def(option.get[ast.def](d));
    auto def_context = option.get[ast.def_id](e.idmap.find(x));

    if (curr_context != def_context) {
        e.sess.span_err(sp, "attempted dynamic environment-capture");
    }

    ret @fold.respan[ast.expr_](sp, ast.expr_path(p, d, a));
}

fn check_for_captures(session.session sess, @ast.crate crate) {
    let fold.ast_fold[env] fld = fold.new_identity_fold[env]();
    fld = @rec( update_env_for_item = bind update_env_for_item(_,_),
                update_env_for_block = bind update_env_for_block(_,_),
                update_env_for_expr = bind update_env_for_expr(_,_),
                fold_expr_path = bind fold_expr_path(_,_,_,_,_)
                with *fld);
    auto idmap = common.new_def_hash[ast.def_id]();
    auto e = rec(current_context = none[ast.def_id], idmap = idmap,
                 sess = sess);
    fold.fold_crate[env](e, fld, crate);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
