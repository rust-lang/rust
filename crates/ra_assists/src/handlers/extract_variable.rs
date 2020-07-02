use ra_syntax::{
    ast::{self, AstNode},
    SyntaxKind::{
        BLOCK_EXPR, BREAK_EXPR, COMMENT, LAMBDA_EXPR, LOOP_EXPR, MATCH_ARM, PATH_EXPR, RETURN_EXPR,
        WHITESPACE,
    },
    SyntaxNode,
};
use stdx::format_to;
use test_utils::mark;

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: extract_variable
//
// Extracts subexpression into a variable.
//
// ```
// fn main() {
//     <|>(1 + 2)<|> * 4;
// }
// ```
// ->
// ```
// fn main() {
//     let $0var_name = (1 + 2);
//     var_name * 4;
// }
// ```
pub(crate) fn extract_variable(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    if ctx.frange.range.is_empty() {
        return None;
    }
    let node = ctx.covering_element();
    if node.kind() == COMMENT {
        mark::hit!(extract_var_in_comment_is_not_applicable);
        return None;
    }
    let expr = node.ancestors().find_map(valid_target_expr)?;
    let (anchor_stmt, wrap_in_block) = anchor_stmt(expr.clone())?;
    let indent = anchor_stmt.prev_sibling_or_token()?.as_token()?.clone();
    if indent.kind() != WHITESPACE {
        return None;
    }
    let target = expr.syntax().text_range();
    acc.add(
        AssistId("extract_variable", AssistKind::RefactorExtract),
        "Extract into variable",
        target,
        move |edit| {
            let field_shorthand = match expr.syntax().parent().and_then(ast::RecordField::cast) {
                Some(field) => field.name_ref(),
                None => None,
            };

            let mut buf = String::new();

            let var_name = match &field_shorthand {
                Some(it) => it.to_string(),
                None => "var_name".to_string(),
            };
            let expr_range = match &field_shorthand {
                Some(it) => it.syntax().text_range().cover(expr.syntax().text_range()),
                None => expr.syntax().text_range(),
            };

            if wrap_in_block {
                format_to!(buf, "{{ let {} = ", var_name);
            } else {
                format_to!(buf, "let {} = ", var_name);
            };
            format_to!(buf, "{}", expr.syntax());

            let full_stmt = ast::ExprStmt::cast(anchor_stmt.clone());
            let is_full_stmt = if let Some(expr_stmt) = &full_stmt {
                Some(expr.syntax().clone()) == expr_stmt.expr().map(|e| e.syntax().clone())
            } else {
                false
            };
            if is_full_stmt {
                mark::hit!(test_extract_var_expr_stmt);
                if full_stmt.unwrap().semicolon_token().is_none() {
                    buf.push_str(";");
                }
                match ctx.config.snippet_cap {
                    Some(cap) => {
                        let snip = buf
                            .replace(&format!("let {}", var_name), &format!("let $0{}", var_name));
                        edit.replace_snippet(cap, expr_range, snip)
                    }
                    None => edit.replace(expr_range, buf),
                }
                return;
            }

            buf.push_str(";");

            // We want to maintain the indent level,
            // but we do not want to duplicate possible
            // extra newlines in the indent block
            let text = indent.text();
            if text.starts_with('\n') {
                buf.push_str("\n");
                buf.push_str(text.trim_start_matches('\n'));
            } else {
                buf.push_str(text);
            }

            edit.replace(expr_range, var_name.clone());
            let offset = anchor_stmt.text_range().start();
            match ctx.config.snippet_cap {
                Some(cap) => {
                    let snip =
                        buf.replace(&format!("let {}", var_name), &format!("let $0{}", var_name));
                    edit.insert_snippet(cap, offset, snip)
                }
                None => edit.insert(offset, buf),
            }

            if wrap_in_block {
                edit.insert(anchor_stmt.text_range().end(), " }");
            }
        },
    )
}

/// Check whether the node is a valid expression which can be extracted to a variable.
/// In general that's true for any expression, but in some cases that would produce invalid code.
fn valid_target_expr(node: SyntaxNode) -> Option<ast::Expr> {
    match node.kind() {
        PATH_EXPR | LOOP_EXPR => None,
        BREAK_EXPR => ast::BreakExpr::cast(node).and_then(|e| e.expr()),
        RETURN_EXPR => ast::ReturnExpr::cast(node).and_then(|e| e.expr()),
        BLOCK_EXPR => {
            ast::BlockExpr::cast(node).filter(|it| it.is_standalone()).map(ast::Expr::from)
        }
        _ => ast::Expr::cast(node),
    }
}

/// Returns the syntax node which will follow the freshly extractd var
/// and a boolean indicating whether we have to wrap it within a { } block
/// to produce correct code.
/// It can be a statement, the last in a block expression or a wanna be block
/// expression like a lambda or match arm.
fn anchor_stmt(expr: ast::Expr) -> Option<(SyntaxNode, bool)> {
    expr.syntax().ancestors().find_map(|node| {
        if let Some(expr) = node.parent().and_then(ast::BlockExpr::cast).and_then(|it| it.expr()) {
            if expr.syntax() == &node {
                mark::hit!(test_extract_var_last_expr);
                return Some((node, false));
            }
        }

        if let Some(parent) = node.parent() {
            if parent.kind() == MATCH_ARM || parent.kind() == LAMBDA_EXPR {
                return Some((node, true));
            }
        }

        if ast::Stmt::cast(node.clone()).is_some() {
            return Some((node, false));
        }

        None
    })
}

#[cfg(test)]
mod tests {
    use test_utils::mark;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

    #[test]
    fn test_extract_var_simple() {
        check_assist(
            extract_variable,
            r#"
fn foo() {
    foo(<|>1 + 1<|>);
}"#,
            r#"
fn foo() {
    let $0var_name = 1 + 1;
    foo(var_name);
}"#,
        );
    }

    #[test]
    fn extract_var_in_comment_is_not_applicable() {
        mark::check!(extract_var_in_comment_is_not_applicable);
        check_assist_not_applicable(extract_variable, "fn main() { 1 + /* <|>comment<|> */ 1; }");
    }

    #[test]
    fn test_extract_var_expr_stmt() {
        mark::check!(test_extract_var_expr_stmt);
        check_assist(
            extract_variable,
            r#"
fn foo() {
    <|>1 + 1<|>;
}"#,
            r#"
fn foo() {
    let $0var_name = 1 + 1;
}"#,
        );
        check_assist(
            extract_variable,
            "
fn foo() {
    <|>{ let x = 0; x }<|>
    something_else();
}",
            "
fn foo() {
    let $0var_name = { let x = 0; x };
    something_else();
}",
        );
    }

    #[test]
    fn test_extract_var_part_of_expr_stmt() {
        check_assist(
            extract_variable,
            "
fn foo() {
    <|>1<|> + 1;
}",
            "
fn foo() {
    let $0var_name = 1;
    var_name + 1;
}",
        );
    }

    #[test]
    fn test_extract_var_last_expr() {
        mark::check!(test_extract_var_last_expr);
        check_assist(
            extract_variable,
            r#"
fn foo() {
    bar(<|>1 + 1<|>)
}
"#,
            r#"
fn foo() {
    let $0var_name = 1 + 1;
    bar(var_name)
}
"#,
        );
        check_assist(
            extract_variable,
            r#"
fn foo() {
    <|>bar(1 + 1)<|>
}
"#,
            r#"
fn foo() {
    let $0var_name = bar(1 + 1);
    var_name
}
"#,
        )
    }

    #[test]
    fn test_extract_var_in_match_arm_no_block() {
        check_assist(
            extract_variable,
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
        true => { let $0var_name = 2 + 2; (var_name, true) }
        _ => (0, false)
    };
}
",
        );
    }

    #[test]
    fn test_extract_var_in_match_arm_with_block() {
        check_assist(
            extract_variable,
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
            let $0var_name = 2 + y;
            (var_name, true)
        }
        _ => (0, false)
    };
}
",
        );
    }

    #[test]
    fn test_extract_var_in_closure_no_block() {
        check_assist(
            extract_variable,
            "
fn main() {
    let lambda = |x: u32| <|>x * 2<|>;
}
",
            "
fn main() {
    let lambda = |x: u32| { let $0var_name = x * 2; var_name };
}
",
        );
    }

    #[test]
    fn test_extract_var_in_closure_with_block() {
        check_assist(
            extract_variable,
            "
fn main() {
    let lambda = |x: u32| { <|>x * 2<|> };
}
",
            "
fn main() {
    let lambda = |x: u32| { let $0var_name = x * 2; var_name };
}
",
        );
    }

    #[test]
    fn test_extract_var_path_simple() {
        check_assist(
            extract_variable,
            "
fn main() {
    let o = <|>Some(true)<|>;
}
",
            "
fn main() {
    let $0var_name = Some(true);
    let o = var_name;
}
",
        );
    }

    #[test]
    fn test_extract_var_path_method() {
        check_assist(
            extract_variable,
            "
fn main() {
    let v = <|>bar.foo()<|>;
}
",
            "
fn main() {
    let $0var_name = bar.foo();
    let v = var_name;
}
",
        );
    }

    #[test]
    fn test_extract_var_return() {
        check_assist(
            extract_variable,
            "
fn foo() -> u32 {
    <|>return 2 + 2<|>;
}
",
            "
fn foo() -> u32 {
    let $0var_name = 2 + 2;
    return var_name;
}
",
        );
    }

    #[test]
    fn test_extract_var_does_not_add_extra_whitespace() {
        check_assist(
            extract_variable,
            "
fn foo() -> u32 {


    <|>return 2 + 2<|>;
}
",
            "
fn foo() -> u32 {


    let $0var_name = 2 + 2;
    return var_name;
}
",
        );

        check_assist(
            extract_variable,
            "
fn foo() -> u32 {

        <|>return 2 + 2<|>;
}
",
            "
fn foo() -> u32 {

        let $0var_name = 2 + 2;
        return var_name;
}
",
        );

        check_assist(
            extract_variable,
            "
fn foo() -> u32 {
    let foo = 1;

    // bar


    <|>return 2 + 2<|>;
}
",
            "
fn foo() -> u32 {
    let foo = 1;

    // bar


    let $0var_name = 2 + 2;
    return var_name;
}
",
        );
    }

    #[test]
    fn test_extract_var_break() {
        check_assist(
            extract_variable,
            "
fn main() {
    let result = loop {
        <|>break 2 + 2<|>;
    };
}
",
            "
fn main() {
    let result = loop {
        let $0var_name = 2 + 2;
        break var_name;
    };
}
",
        );
    }

    #[test]
    fn test_extract_var_for_cast() {
        check_assist(
            extract_variable,
            "
fn main() {
    let v = <|>0f32 as u32<|>;
}
",
            "
fn main() {
    let $0var_name = 0f32 as u32;
    let v = var_name;
}
",
        );
    }

    #[test]
    fn extract_var_field_shorthand() {
        check_assist(
            extract_variable,
            r#"
struct S {
    foo: i32
}

fn main() {
    S { foo: <|>1 + 1<|> }
}
"#,
            r#"
struct S {
    foo: i32
}

fn main() {
    let $0foo = 1 + 1;
    S { foo }
}
"#,
        )
    }

    #[test]
    fn test_extract_var_for_return_not_applicable() {
        check_assist_not_applicable(extract_variable, "fn foo() { <|>return<|>; } ");
    }

    #[test]
    fn test_extract_var_for_break_not_applicable() {
        check_assist_not_applicable(extract_variable, "fn main() { loop { <|>break<|>; }; }");
    }

    // FIXME: This is not quite correct, but good enough(tm) for the sorting heuristic
    #[test]
    fn extract_var_target() {
        check_assist_target(extract_variable, "fn foo() -> u32 { <|>return 2 + 2<|>; }", "2 + 2");

        check_assist_target(
            extract_variable,
            "
fn main() {
    let x = true;
    let tuple = match x {
        true => (<|>2 + 2<|>, true)
        _ => (0, false)
    };
}
",
            "2 + 2",
        );
    }
}
