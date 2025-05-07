//! Things which exist to solve practical issues, but which shouldn't exist.
//!
//! Please avoid adding new usages of the functions in this module

use parser::Edition;

use crate::{AstNode, ast};

pub fn parse_expr_from_str(s: &str, edition: Edition) -> Option<ast::Expr> {
    let s = s.trim();

    let file = ast::SourceFile::parse(
        // Need a newline because the text may contain line comments.
        &format!("const _: () = ({s}\n);"),
        edition,
    );
    let expr = file.syntax_node().descendants().find_map(ast::ParenExpr::cast)?;
    // Can't check the text because the original text may contain whitespace and comments.
    // Wrap in parentheses to better allow for verification. Of course, the real fix is
    // to get rid of this hack.
    expr.expr()
}
