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
    return match stmt.node {
        ast::stmt_decl(d, _) => {
            match d.node {
                ast::decl_local(_) => true,
                ast::decl_item(_) => false
            }
        }
        ast::stmt_expr(e, _) => { expr_requires_semi_to_be_stmt(e) }
        ast::stmt_semi(*) => { false }
        ast::stmt_mac(*) => { false }
    }
}
