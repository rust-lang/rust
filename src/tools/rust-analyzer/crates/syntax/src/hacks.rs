//! Things which exist to solve practical issues, but which shouldn't exist.
//!
//! Please avoid adding new usages of the functions in this module

use crate::{ast, AstNode};

pub fn parse_expr_from_str(s: &str) -> Option<ast::Expr> {
    let s = s.trim();
    let file = ast::SourceFile::parse(&format!("const _: () = {};", s));
    let expr = file.syntax_node().descendants().find_map(ast::Expr::cast)?;
    if expr.syntax().text() != s {
        return None;
    }
    Some(expr)
}
