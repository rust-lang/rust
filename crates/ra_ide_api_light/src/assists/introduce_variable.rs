use ra_syntax::{
    ast::{self, AstNode},
    SyntaxKind::WHITESPACE, SyntaxKind::MATCH_ARM, SyntaxKind::LAMBDA_EXPR,
    SyntaxNode, TextUnit,
};

use crate::assists::{AssistCtx, Assist};

pub fn introduce_variable<'a>(ctx: AssistCtx) -> Option<Assist> {
    let node = ctx.covering_node();
    let expr = node.ancestors().filter_map(ast::Expr::cast).next()?;

    let (anchor_stmt, wrap_in_block) = anchor_stmt(expr)?;
    let indent = anchor_stmt.prev_sibling()?;
    if indent.kind() != WHITESPACE {
        return None;
    }
    ctx.build("introduce variable", move |edit| {
        let mut buf = String::new();

        let cursor_offset = if wrap_in_block {
            buf.push_str("{ let var_name = ");
            TextUnit::of_str("{ let ")
        } else {
            buf.push_str("let var_name = ");
            TextUnit::of_str("let ")
        };

        expr.syntax().text().push_to(&mut buf);
        let full_stmt = ast::ExprStmt::cast(anchor_stmt);
        let is_full_stmt = if let Some(expr_stmt) = full_stmt {
            Some(expr.syntax()) == expr_stmt.expr().map(|e| e.syntax())
        } else {
            false
        };
        if is_full_stmt {
            if !full_stmt.unwrap().has_semi() {
                buf.push_str(";");
            }
            edit.replace(expr.syntax().range(), buf);
        } else {
            buf.push_str(";");
            indent.text().push_to(&mut buf);
            edit.replace(expr.syntax().range(), "var_name".to_string());
            edit.insert(anchor_stmt.range().start(), buf);
            if wrap_in_block {
                edit.insert(anchor_stmt.range().end(), " }");
            }
        }
        edit.set_cursor(anchor_stmt.range().start() + cursor_offset);
    })
}

/// Returns the syntax node which will follow the freshly introduced var
/// and a boolean indicating whether we have to wrap it within a { } block
/// to produce correct code.
/// It can be a statement, the last in a block expression or a wanna be block
/// expression like a lamba or match arm.
fn anchor_stmt(expr: &ast::Expr) -> Option<(&SyntaxNode, bool)> {
    expr.syntax().ancestors().find_map(|node| {
        if ast::Stmt::cast(node).is_some() {
            return Some((node, false));
        }

        if let Some(expr) = node
            .parent()
            .and_then(ast::Block::cast)
            .and_then(|it| it.expr())
        {
            if expr.syntax() == node {
                return Some((node, false));
            }
        }

        if let Some(parent) = node.parent() {
            if parent.kind() == MATCH_ARM
                || parent.kind() == LAMBDA_EXPR
            {
                return Some((node, true));
            }
        }

        None
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

    #[test]
    fn test_introduce_var_in_match_arm_no_block() {
        check_assist_range(
            introduce_variable,
            "
fn main() {
    let x = true;
    let tuple = match x {
        true => (<|>2 + 2<|>, true)
        _ => (0, false)
    };
}
",
            "
fn main() {
    let x = true;
    let tuple = match x {
        true => { let <|>var_name = 2 + 2; (var_name, true) }
        _ => (0, false)
    };
}
",
        );
    }

    #[test]
    fn test_introduce_var_in_match_arm_with_block() {
        check_assist_range(
            introduce_variable,
            "
fn main() {
    let x = true;
    let tuple = match x {
        true => {
            let y = 1;
            (<|>2 + y<|>, true)
        }
        _ => (0, false)
    };
}
",
            "
fn main() {
    let x = true;
    let tuple = match x {
        true => {
            let y = 1;
            let <|>var_name = 2 + y;
            (var_name, true)
        }
        _ => (0, false)
    };
}
",
        );
    }

    #[test]
    fn test_introduce_var_in_closure_no_block() {
        check_assist_range(
            introduce_variable,
            "
fn main() {
    let lambda = |x: u32| <|>x * 2<|>;
}
",
            "
fn main() {
    let lambda = |x: u32| { let <|>var_name = x * 2; var_name };
}
",
        );
    }

    #[test]
    fn test_introduce_var_in_closure_with_block() {
        check_assist_range(
            introduce_variable,
            "
fn main() {
    let lambda = |x: u32| { <|>x * 2<|> };
}
",
            "
fn main() {
    let lambda = |x: u32| { let <|>var_name = x * 2; var_name };
}
",
        );
    }
}
