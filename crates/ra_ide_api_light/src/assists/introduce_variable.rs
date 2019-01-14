use ra_syntax::{
    ast::{self, AstNode},
    SyntaxKind::{WHITESPACE, BLOCK_EXPR},
    SyntaxNode, TextUnit,
};

use crate::assists::{AssistCtx, Assist};

pub fn introduce_variable<'a>(ctx: AssistCtx) -> Option<Assist> {
    let node = ctx.covering_node();
    let expr = node.ancestors().filter_map(ast::Expr::cast).next()?;

    let anchor_stmt = anchor_stmt(expr)?;
    let indent = anchor_stmt.prev_sibling()?;
    if indent.kind() != WHITESPACE {
        return None;
    }
    ctx.build("introduce variable", move |edit| {
        let mut buf = String::new();

        buf.push_str("let var_name = ");
        expr.syntax().text().push_to(&mut buf);
        let is_full_stmt = if let Some(expr_stmt) = ast::ExprStmt::cast(anchor_stmt) {
            Some(expr.syntax()) == expr_stmt.expr().map(|e| e.syntax())
        } else {
            false
        };
        if is_full_stmt {
            if expr.syntax().kind() == BLOCK_EXPR {
                buf.push_str(";");
            }
            edit.replace(expr.syntax().range(), buf);
        } else {
            buf.push_str(";");
            indent.text().push_to(&mut buf);
            edit.replace(expr.syntax().range(), "var_name".to_string());
            edit.insert(anchor_stmt.range().start(), buf);
        }
        edit.set_cursor(anchor_stmt.range().start() + TextUnit::of_str("let "));
    })
}

/// Statement or last in the block expression, which will follow
/// the freshly introduced var.
fn anchor_stmt(expr: &ast::Expr) -> Option<&SyntaxNode> {
    expr.syntax().ancestors().find(|&node| {
        if ast::Stmt::cast(node).is_some() {
            return true;
        }
        if let Some(expr) = node
            .parent()
            .and_then(ast::Block::cast)
            .and_then(|it| it.expr())
        {
            if expr.syntax() == node {
                return true;
            }
        }
        false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assists::check_assist_range;

    #[test]
    fn test_introduce_var_simple() {
        check_assist_range(
            introduce_variable,
            "
fn foo() {
    foo(<|>1 + 1<|>);
}",
            "
fn foo() {
    let <|>var_name = 1 + 1;
    foo(var_name);
}",
        );
    }

    #[test]
    fn test_introduce_var_expr_stmt() {
        check_assist_range(
            introduce_variable,
            "
fn foo() {
    <|>1 + 1<|>;
}",
            "
fn foo() {
    let <|>var_name = 1 + 1;
}",
        );
    }

    #[test]
    fn test_introduce_var_part_of_expr_stmt() {
        check_assist_range(
            introduce_variable,
            "
fn foo() {
    <|>1<|> + 1;
}",
            "
fn foo() {
    let <|>var_name = 1;
    var_name + 1;
}",
        );
    }

    #[test]
    fn test_introduce_var_last_expr() {
        check_assist_range(
            introduce_variable,
            "
fn foo() {
    bar(<|>1 + 1<|>)
}",
            "
fn foo() {
    let <|>var_name = 1 + 1;
    bar(var_name)
}",
        );
    }

    #[test]
    fn test_introduce_var_last_full_expr() {
        check_assist_range(
            introduce_variable,
            "
fn foo() {
    <|>bar(1 + 1)<|>
}",
            "
fn foo() {
    let <|>var_name = bar(1 + 1);
    var_name
}",
        );
    }

    #[test]
    fn test_introduce_var_block_expr_second_to_last() {
        check_assist_range(
            introduce_variable,
            "
fn foo() {
    <|>{ let x = 0; x }<|>
    something_else();
}",
            "
fn foo() {
    let <|>var_name = { let x = 0; x };
    something_else();
}",
        );
    }
}
