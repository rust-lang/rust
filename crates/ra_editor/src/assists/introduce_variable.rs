use ra_text_edit::TextEditBuilder;
use ra_syntax::{
    algo::{find_covering_node},
    ast::{self, AstNode},
    SourceFileNode,
    SyntaxKind::{WHITESPACE},
    SyntaxNodeRef, TextRange, TextUnit,
};

use crate::assists::LocalEdit;

pub fn introduce_variable<'a>(
    file: &'a SourceFileNode,
    range: TextRange,
) -> Option<impl FnOnce() -> LocalEdit + 'a> {
    let node = find_covering_node(file.syntax(), range);
    let expr = node.ancestors().filter_map(ast::Expr::cast).next()?;

    let anchor_stmt = anchor_stmt(expr)?;
    let indent = anchor_stmt.prev_sibling()?;
    if indent.kind() != WHITESPACE {
        return None;
    }
    return Some(move || {
        let mut buf = String::new();
        let mut edit = TextEditBuilder::new();

        buf.push_str("let var_name = ");
        expr.syntax().text().push_to(&mut buf);
        let is_full_stmt = if let Some(expr_stmt) = ast::ExprStmt::cast(anchor_stmt) {
            Some(expr.syntax()) == expr_stmt.expr().map(|e| e.syntax())
        } else {
            false
        };
        if is_full_stmt {
            edit.replace(expr.syntax().range(), buf);
        } else {
            buf.push_str(";");
            indent.text().push_to(&mut buf);
            edit.replace(expr.syntax().range(), "var_name".to_string());
            edit.insert(anchor_stmt.range().start(), buf);
        }
        let cursor_position = anchor_stmt.range().start() + TextUnit::of_str("let ");
        LocalEdit {
            label: "introduce variable".to_string(),
            edit: edit.finish(),
            cursor_position: Some(cursor_position),
        }
    });

    /// Statement or last in the block expression, which will follow
    /// the freshly introduced var.
    fn anchor_stmt(expr: ast::Expr) -> Option<SyntaxNodeRef> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::check_action_range;

    #[test]
    fn test_introduce_var_simple() {
        check_action_range(
            "
fn foo() {
    foo(<|>1 + 1<|>);
}",
            "
fn foo() {
    let <|>var_name = 1 + 1;
    foo(var_name);
}",
            |file, range| introduce_variable(file, range).map(|f| f()),
        );
    }

    #[test]
    fn test_introduce_var_expr_stmt() {
        check_action_range(
            "
fn foo() {
    <|>1 + 1<|>;
}",
            "
fn foo() {
    let <|>var_name = 1 + 1;
}",
            |file, range| introduce_variable(file, range).map(|f| f()),
        );
    }

    #[test]
    fn test_introduce_var_part_of_expr_stmt() {
        check_action_range(
            "
fn foo() {
    <|>1<|> + 1;
}",
            "
fn foo() {
    let <|>var_name = 1;
    var_name + 1;
}",
            |file, range| introduce_variable(file, range).map(|f| f()),
        );
    }

    #[test]
    fn test_introduce_var_last_expr() {
        check_action_range(
            "
fn foo() {
    bar(<|>1 + 1<|>)
}",
            "
fn foo() {
    let <|>var_name = 1 + 1;
    bar(var_name)
}",
            |file, range| introduce_variable(file, range).map(|f| f()),
        );
    }

    #[test]
    fn test_introduce_var_last_full_expr() {
        check_action_range(
            "
fn foo() {
    <|>bar(1 + 1)<|>
}",
            "
fn foo() {
    let <|>var_name = bar(1 + 1);
    var_name
}",
            |file, range| introduce_variable(file, range).map(|f| f()),
        );
    }

}
