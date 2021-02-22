use stdx::format_to;
use syntax::{
    ast::{self, AstNode},
    SyntaxKind::{
        BLOCK_EXPR, BREAK_EXPR, CLOSURE_EXPR, COMMENT, LOOP_EXPR, MATCH_ARM, PATH_EXPR, RETURN_EXPR,
    },
    SyntaxNode,
};
use test_utils::mark;

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: extract_variable
//
// Extracts subexpression into a variable.
//
// ```
// fn main() {
//     $0(1 + 2)$0 * 4;
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
    let to_extract = node.ancestors().find_map(valid_target_expr)?;
    let anchor = Anchor::from(&to_extract)?;
    let indent = anchor.syntax().prev_sibling_or_token()?.as_token()?.clone();
    let target = to_extract.syntax().text_range();
    acc.add(
        AssistId("extract_variable", AssistKind::RefactorExtract),
        "Extract into variable",
        target,
        move |edit| {
            let field_shorthand =
                match to_extract.syntax().parent().and_then(ast::RecordExprField::cast) {
                    Some(field) => field.name_ref(),
                    None => None,
                };

            let mut buf = String::new();

            let var_name = match &field_shorthand {
                Some(it) => it.to_string(),
                None => "var_name".to_string(),
            };
            let expr_range = match &field_shorthand {
                Some(it) => it.syntax().text_range().cover(to_extract.syntax().text_range()),
                None => to_extract.syntax().text_range(),
            };

            if let Anchor::WrapInBlock(_) = anchor {
                format_to!(buf, "{{ let {} = ", var_name);
            } else {
                format_to!(buf, "let {} = ", var_name);
            };
            format_to!(buf, "{}", to_extract.syntax());

            if let Anchor::Replace(stmt) = anchor {
                mark::hit!(test_extract_var_expr_stmt);
                if stmt.semicolon_token().is_none() {
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
                buf.push('\n');
                buf.push_str(text.trim_start_matches('\n'));
            } else {
                buf.push_str(text);
            }

            edit.replace(expr_range, var_name.clone());
            let offset = anchor.syntax().text_range().start();
            match ctx.config.snippet_cap {
                Some(cap) => {
                    let snip =
                        buf.replace(&format!("let {}", var_name), &format!("let $0{}", var_name));
                    edit.insert_snippet(cap, offset, snip)
                }
                None => edit.insert(offset, buf),
            }

            if let Anchor::WrapInBlock(_) = anchor {
                edit.insert(anchor.syntax().text_range().end(), " }");
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

enum Anchor {
    Before(SyntaxNode),
    Replace(ast::ExprStmt),
    WrapInBlock(SyntaxNode),
}

impl Anchor {
    fn from(to_extract: &ast::Expr) -> Option<Anchor> {
        to_extract.syntax().ancestors().find_map(|node| {
            if let Some(expr) =
                node.parent().and_then(ast::BlockExpr::cast).and_then(|it| it.tail_expr())
            {
                if expr.syntax() == &node {
                    mark::hit!(test_extract_var_last_expr);
                    return Some(Anchor::Before(node));
                }
            }

            if let Some(parent) = node.parent() {
                if parent.kind() == MATCH_ARM || parent.kind() == CLOSURE_EXPR {
                    return Some(Anchor::WrapInBlock(node));
                }
            }

            if let Some(stmt) = ast::Stmt::cast(node.clone()) {
                if let ast::Stmt::ExprStmt(stmt) = stmt {
                    if stmt.expr().as_ref() == Some(to_extract) {
                        return Some(Anchor::Replace(stmt));
                    }
                }
                return Some(Anchor::Before(node));
            }
            None
        })
    }

    fn syntax(&self) -> &SyntaxNode {
        match self {
            Anchor::Before(it) | Anchor::WrapInBlock(it) => it,
            Anchor::Replace(stmt) => stmt.syntax(),
        }
    }
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
    foo($01 + 1$0);
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
        check_assist_not_applicable(extract_variable, "fn main() { 1 + /* $0comment$0 */ 1; }");
    }

    #[test]
    fn test_extract_var_expr_stmt() {
        mark::check!(test_extract_var_expr_stmt);
        check_assist(
            extract_variable,
            r#"
fn foo() {
    $01 + 1$0;
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
    $0{ let x = 0; x }$0
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
    $01$0 + 1;
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
    bar($01 + 1$0)
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
    $0bar(1 + 1)$0
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
        true => ($02 + 2$0, true)
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
            ($02 + y$0, true)
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
    let lambda = |x: u32| $0x * 2$0;
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
    let lambda = |x: u32| { $0x * 2$0 };
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
    let o = $0Some(true)$0;
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
    let v = $0bar.foo()$0;
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
    $0return 2 + 2$0;
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


    $0return 2 + 2$0;
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

        $0return 2 + 2$0;
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


    $0return 2 + 2$0;
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
        $0break 2 + 2$0;
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
    let v = $00f32 as u32$0;
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
    S { foo: $01 + 1$0 }
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
        check_assist_not_applicable(extract_variable, "fn foo() { $0return$0; } ");
    }

    #[test]
    fn test_extract_var_for_break_not_applicable() {
        check_assist_not_applicable(extract_variable, "fn main() { loop { $0break$0; }; }");
    }

    // FIXME: This is not quite correct, but good enough(tm) for the sorting heuristic
    #[test]
    fn extract_var_target() {
        check_assist_target(extract_variable, "fn foo() -> u32 { $0return 2 + 2$0; }", "2 + 2");

        check_assist_target(
            extract_variable,
            "
fn main() {
    let x = true;
    let tuple = match x {
        true => ($02 + 2$0, true)
        _ => (0, false)
    };
}
",
            "2 + 2",
        );
    }
}
