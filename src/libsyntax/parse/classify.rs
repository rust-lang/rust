/*
  Predicates on exprs and stmts that the pretty-printer and parser use
 */

use ast_util::operator_prec;

fn expr_requires_semi_to_be_stmt(e: @ast::expr) -> bool {
    match e.node {
      ast::expr_if(*) | ast::expr_match(*) | ast::expr_block(_)
      | ast::expr_while(*) | ast::expr_loop(*)
      | ast::expr_call(_, _, true) => false,
      _ => true
    }
}

fn expr_is_simple_block(e: @ast::expr) -> bool {
    match e.node {
      ast::expr_block({node: {rules: ast::default_blk, _}, _}) => true,
      _ => false
    }
}

fn stmt_ends_with_semi(stmt: ast::stmt) -> bool {
    match stmt.node {
      ast::stmt_decl(d, _) => {
        return match d.node {
              ast::decl_local(_) => true,
              ast::decl_item(_) => false
            }
      }
      ast::stmt_expr(e, _) => {
        return expr_requires_semi_to_be_stmt(e);
      }
      ast::stmt_semi(*) => {
        return false;
      }
    }
}

fn need_parens(expr: @ast::expr, outer_prec: uint) -> bool {
    match expr.node {
      ast::expr_binary(op, _, _) => operator_prec(op) < outer_prec,
      ast::expr_cast(_, _) => parse::prec::as_prec < outer_prec,
      // This may be too conservative in some cases
      ast::expr_assign(_, _) => true,
      ast::expr_move(_, _) => true,
      ast::expr_swap(_, _) => true,
      ast::expr_assign_op(_, _, _) => true,
      ast::expr_ret(_) => true,
      ast::expr_assert(_) => true,
      ast::expr_log(_, _, _) => true,
      _ => !parse::classify::expr_requires_semi_to_be_stmt(expr)
    }
}

fn ends_in_lit_int(ex: @ast::expr) -> bool {
    match ex.node {
      ast::expr_lit(node) => match node {
        @{node: ast::lit_int(_, ast::ty_i), _}
        | @{node: ast::lit_int_unsuffixed(_), _} => true,
        _ => false
      },
      ast::expr_binary(_, _, sub) | ast::expr_unary(_, sub) |
      ast::expr_move(_, sub) | ast::expr_copy(sub) |
      ast::expr_assign(_, sub) |
      ast::expr_assign_op(_, _, sub) | ast::expr_swap(_, sub) |
      ast::expr_log(_, _, sub) | ast::expr_assert(sub) => {
        ends_in_lit_int(sub)
      }
      ast::expr_fail(osub) | ast::expr_ret(osub) => match osub {
        Some(ex) => ends_in_lit_int(ex),
        _ => false
      },
      _ => false
    }
}
