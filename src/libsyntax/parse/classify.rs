/*
  Predicates on exprs and stmts that the pretty-printer and parser use
 */

import ast_util::operator_prec;

fn expr_requires_semi_to_be_stmt(e: @ast::expr) -> bool {
    alt e.node {
      ast::expr_if(_, _, _) | ast::expr_if_check(_, _, _)
      | ast::expr_alt(_, _, _) | ast::expr_block(_)
      | ast::expr_while(_, _) | ast::expr_loop(_)
      | ast::expr_call(_, _, true) {
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

fn need_parens(expr: @ast::expr, outer_prec: uint) -> bool {
    alt expr.node {
      ast::expr_binary(op, _, _) { operator_prec(op) < outer_prec }
      ast::expr_cast(_, _) { parse::prec::as_prec < outer_prec }
      // This may be too conservative in some cases
      ast::expr_assign(_, _) { true }
      ast::expr_move(_, _) { true }
      ast::expr_swap(_, _) { true }
      ast::expr_assign_op(_, _, _) { true }
      ast::expr_ret(_) { true }
      ast::expr_assert(_) { true }
      ast::expr_check(_, _) { true }
      ast::expr_log(_, _, _) { true }
      _ { !parse::classify::expr_requires_semi_to_be_stmt(expr) }
    }
}

fn ends_in_lit_int(ex: @ast::expr) -> bool {
    alt ex.node {
      ast::expr_lit(node) {
        alt node {
          @{node: ast::lit_int(_, ast::ty_i), _} |
          @{node: ast::lit_int_unsuffixed(_), _}
          { true }
          _ { false }
        }
      }
      ast::expr_binary(_, _, sub) | ast::expr_unary(_, sub) |
      ast::expr_move(_, sub) | ast::expr_copy(sub) |
      ast::expr_assign(_, sub) |
      ast::expr_assign_op(_, _, sub) | ast::expr_swap(_, sub) |
      ast::expr_log(_, _, sub) | ast::expr_assert(sub) |
      ast::expr_check(_, sub) { ends_in_lit_int(sub) }
      ast::expr_fail(osub) | ast::expr_ret(osub) {
        alt osub {
          some(ex) { ends_in_lit_int(ex) }
          _ { false }
        }
      }
      _ { false }
    }
}
