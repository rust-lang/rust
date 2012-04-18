// FIXME: There are a bunch of similar functions in pprust that
// likely belong here

fn expr_requires_semi_to_be_stmt(e: @ast::expr) -> bool {
    alt e.node {
      ast::expr_if(_, _, _) | ast::expr_if_check(_, _, _)
      | ast::expr_alt(_, _, _) | ast::expr_block(_)
      | ast::expr_do_while(_, _) | ast::expr_while(_, _)
      | ast::expr_loop(_) | ast::expr_call(_, _, true) {
        false
      }
      _ { true }
    }
}

fn stmt_ends_with_semi(stmt: ast::stmt) -> bool {
    alt stmt.node {
      ast::stmt_decl(d, _) {
        ret alt d.node {
              ast::decl_local(_) { true }
              ast::decl_item(_) { false }
            }
      }
      ast::stmt_expr(e, _) {
        ret expr_requires_semi_to_be_stmt(e);
      }
      ast::stmt_semi(e, _) {
        ret false;
      }
    }
}
