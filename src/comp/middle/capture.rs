import syntax::{ast, ast_util};
import std::map;

export capture_mode;
export capture_var;
export capture_map;
export check_capture_clause;
export compute_capture_vars;
export cap_copy;
export cap_move;
export cap_drop;
export cap_ref;

tag capture_mode {
    cap_copy; //< Copy the value into the closure.
    cap_move; //< Move the value into the closure.
    cap_drop; //< Drop value after creating closure.
    cap_ref;  //< Reference directly from parent stack frame (block fn).
}

type capture_var = {
    def: ast::def,     //< The variable being accessed free.
    mode: capture_mode //< How is the variable being accessed.
};

type capture_map = map::hashmap<ast::def_id, capture_var>;

// checks the capture clause for a fn_expr() and issues warnings or
// errors for any irregularities which we identify.
fn check_capture_clause(tcx: ty::ctxt,
                        fn_expr_id: ast::node_id,
                        fn_proto: ast::proto,
                        cap_clause: ast::capture_clause) {
    let freevars = freevars::get_freevars(tcx, fn_expr_id);
    let seen_defs = map::new_int_hash();

    let check_capture_item = fn@(&&cap_item: @ast::capture_item) {
        let cap_def = tcx.def_map.get(cap_item.id);
        if !vec::any(*freevars, {|fv| fv.def == cap_def}) {
            tcx.sess.span_warn(
                cap_item.span,
                #fmt("Captured variable '%s' not used in closure",
                     cap_item.name));
        }

        let cap_def_id = ast_util::def_id_of_def(cap_def).node;
        if !seen_defs.insert(cap_def_id, ()) {
            tcx.sess.span_err(
                cap_item.span,
                #fmt("Variable '%s' captured more than once",
                     cap_item.name));
        }
    };

    let check_not_upvar = fn@(&&cap_item: @ast::capture_item) {
        alt tcx.def_map.get(cap_item.id) {
          ast::def_upvar(_, _, _) {
            tcx.sess.span_err(
                cap_item.span,
                #fmt("Upvars (like '%s') cannot be moved into a closure",
                     cap_item.name));
          }
          _ {}
        }
    };

    let check_block_captures = fn@(v: [@ast::capture_item]) {
        if check vec::is_not_empty(v) {
            let cap_item0 = vec::head(v);
            tcx.sess.span_err(
                cap_item0.span,
                "Cannot capture values explicitly with a block closure");
        }
    };

    alt fn_proto {
      ast::proto_block. {
        check_block_captures(cap_clause.copies);
        check_block_captures(cap_clause.moves);
      }
      ast::proto_bare. | ast::proto_shared. | ast::proto_send. {
        vec::iter(cap_clause.copies, check_capture_item);
        vec::iter(cap_clause.moves, check_capture_item);
        vec::iter(cap_clause.moves, check_not_upvar);
      }
    }
}

fn compute_capture_vars(tcx: ty::ctxt,
                        fn_expr_id: ast::node_id,
                        fn_proto: ast::proto,
                        cap_clause: ast::capture_clause) -> [capture_var] {
    let freevars = freevars::get_freevars(tcx, fn_expr_id);
    let cap_map = map::new_int_hash();

    vec::iter(cap_clause.copies) { |cap_item|
        let cap_def = tcx.def_map.get(cap_item.id);
        let cap_def_id = ast_util::def_id_of_def(cap_def).node;
        if vec::any(*freevars, {|fv| fv.def == cap_def}) {
            cap_map.insert(cap_def_id, { def:cap_def, mode:cap_copy });
        }
    }

    vec::iter(cap_clause.moves) { |cap_item|
        let cap_def = tcx.def_map.get(cap_item.id);
        let cap_def_id = ast_util::def_id_of_def(cap_def).node;
        if vec::any(*freevars, {|fv| fv.def == cap_def}) {
            cap_map.insert(cap_def_id, { def:cap_def, mode:cap_move });
        } else {
            cap_map.insert(cap_def_id, { def:cap_def, mode:cap_drop });
        }
    }

    let implicit_mode = alt fn_proto {
      ast::proto_block. { cap_ref }
      ast::proto_bare. | ast::proto_shared. | ast::proto_send. { cap_copy }
    };

    vec::iter(*freevars) { |fvar|
        let fvar_def_id = ast_util::def_id_of_def(fvar.def).node;
        alt cap_map.find(fvar_def_id) {
          option::some(_) { /* was explicitly named, do nothing */ }
          option::none. {
            cap_map.insert(fvar_def_id, {def:fvar.def, mode:implicit_mode});
          }
        }
    }

    let result = [];
    cap_map.values { |cap_var| result += [cap_var]; }
    ret result;
}
