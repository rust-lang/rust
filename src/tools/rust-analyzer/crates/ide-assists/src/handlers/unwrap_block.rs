use syntax::{
    AstNode, SyntaxKind, T, TextRange,
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make,
    },
};

use crate::{AssistContext, AssistId, Assists};

// Assist: unwrap_block
//
// This assist removes if...else, for, while and loop control statements to just keep the body.
//
// ```
// fn foo() {
//     if true {$0
//         println!("foo");
//     }
// }
// ```
// ->
// ```
// fn foo() {
//     println!("foo");
// }
// ```
pub(crate) fn unwrap_block(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let assist_id = AssistId::refactor_rewrite("unwrap_block");
    let assist_label = "Unwrap block";
    let l_curly_token = ctx.find_token_syntax_at_offset(T!['{'])?;
    let mut block = ast::BlockExpr::cast(l_curly_token.parent_ancestors().nth(1)?)?;
    let target = block.syntax().text_range();
    let mut parent = block.syntax().parent()?;
    if ast::MatchArm::can_cast(parent.kind()) {
        parent = parent.ancestors().find(|it| ast::MatchExpr::can_cast(it.kind()))?
    }

    let kind = parent.kind();
    if matches!(kind, SyntaxKind::STMT_LIST | SyntaxKind::EXPR_STMT) {
        acc.add(assist_id, assist_label, target, |builder| {
            builder.replace(block.syntax().text_range(), update_expr_string(block.to_string()));
        })
    } else if matches!(kind, SyntaxKind::LET_STMT) {
        let parent = ast::LetStmt::cast(parent)?;
        let pattern = ast::Pat::cast(parent.syntax().first_child()?)?;
        let ty = parent.ty();
        let list = block.stmt_list()?;
        let replaced = match list.syntax().last_child() {
            Some(last) => {
                let stmts: Vec<ast::Stmt> = list.statements().collect();
                let initializer = ast::Expr::cast(last)?;
                let let_stmt = make::let_stmt(pattern, ty, Some(initializer));
                if !stmts.is_empty() {
                    let block = make::block_expr(stmts, None);
                    format!("{}\n    {}", update_expr_string(block.to_string()), let_stmt)
                } else {
                    let_stmt.to_string()
                }
            }
            None => {
                let empty_tuple = make::ext::expr_unit();
                make::let_stmt(pattern, ty, Some(empty_tuple)).to_string()
            }
        };
        acc.add(assist_id, assist_label, target, |builder| {
            builder.replace(parent.syntax().text_range(), replaced);
        })
    } else {
        let parent = ast::Expr::cast(parent)?;
        match parent.clone() {
            ast::Expr::ForExpr(_) | ast::Expr::WhileExpr(_) | ast::Expr::LoopExpr(_) => (),
            ast::Expr::MatchExpr(_) => block = block.dedent(IndentLevel(1)),
            ast::Expr::IfExpr(if_expr) => {
                let then_branch = if_expr.then_branch()?;
                if then_branch == block {
                    if let Some(ancestor) = if_expr.syntax().parent().and_then(ast::IfExpr::cast) {
                        // For `else if` blocks
                        let ancestor_then_branch = ancestor.then_branch()?;

                        return acc.add(assist_id, assist_label, target, |edit| {
                            let range_to_del_else_if = TextRange::new(
                                ancestor_then_branch.syntax().text_range().end(),
                                l_curly_token.text_range().start(),
                            );
                            let range_to_del_rest = TextRange::new(
                                then_branch.syntax().text_range().end(),
                                if_expr.syntax().text_range().end(),
                            );

                            edit.delete(range_to_del_rest);
                            edit.delete(range_to_del_else_if);
                            edit.replace(
                                target,
                                update_expr_string_without_newline(then_branch.to_string()),
                            );
                        });
                    }
                } else {
                    return acc.add(assist_id, assist_label, target, |edit| {
                        let range_to_del = TextRange::new(
                            then_branch.syntax().text_range().end(),
                            l_curly_token.text_range().start(),
                        );

                        edit.delete(range_to_del);
                        edit.replace(target, update_expr_string_without_newline(block.to_string()));
                    });
                }
            }
            _ => return None,
        };

        acc.add(assist_id, assist_label, target, |builder| {
            builder.replace(parent.syntax().text_range(), update_expr_string(block.to_string()));
        })
    }
}

fn update_expr_string(expr_string: String) -> String {
    update_expr_string_with_pat(expr_string, &[' ', '\n'])
}

fn update_expr_string_without_newline(expr_string: String) -> String {
    update_expr_string_with_pat(expr_string, &[' '])
}

fn update_expr_string_with_pat(expr_str: String, whitespace_pat: &[char]) -> String {
    // Remove leading whitespace, index to remove the leading '{',
    // then continue to remove leading whitespace.
    // We cannot assume the `{` is the first character because there are block modifiers
    // (`unsafe`, `async` etc.).
    let after_open_brace_index = expr_str.find('{').map_or(0, |it| it + 1);
    let expr_str = expr_str[after_open_brace_index..].trim_start_matches(whitespace_pat);

    // Remove trailing whitespace, index [..expr_str.len() - 1] to remove the trailing '}',
    // then continue to remove trailing whitespace.
    let expr_str = expr_str.trim_end_matches(whitespace_pat);
    let expr_str = expr_str[..expr_str.len() - 1].trim_end_matches(whitespace_pat);

    expr_str
        .lines()
        .map(|line| line.replacen("    ", "", 1)) // Delete indentation
        .collect::<Vec<String>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn unwrap_tail_expr_block() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    $0{
        92
    }
}
"#,
            r#"
fn main() {
    92
}
"#,
        )
    }

    #[test]
    fn unwrap_stmt_expr_block() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    $0{
        92;
    }
    ()
}
"#,
            r#"
fn main() {
    92;
    ()
}
"#,
        );
        // Pedantically, we should add an `;` here...
        check_assist(
            unwrap_block,
            r#"
fn main() {
    $0{
        92
    }
    ()
}
"#,
            r#"
fn main() {
    92
    ()
}
"#,
        );
    }

    #[test]
    fn simple_if() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    bar();
    if true {$0
        foo();

        // comment
        bar();
    } else {
        println!("bar");
    }
}
"#,
            r#"
fn main() {
    bar();
    foo();

    // comment
    bar();
}
"#,
        );
    }

    #[test]
    fn simple_if_else() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    bar();
    if true {
        foo();

        // comment
        bar();
    } else {$0
        println!("bar");
    }
}
"#,
            r#"
fn main() {
    bar();
    if true {
        foo();

        // comment
        bar();
    }
    println!("bar");
}
"#,
        );
    }

    #[test]
    fn simple_if_else_if() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    // bar();
    if true {
        println!("true");

        // comment
        // bar();
    } else if false {$0
        println!("bar");
    } else {
        println!("foo");
    }
}
"#,
            r#"
fn main() {
    // bar();
    if true {
        println!("true");

        // comment
        // bar();
    }
    println!("bar");
}
"#,
        );
    }

    #[test]
    fn simple_if_else_if_nested() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    // bar();
    if true {
        println!("true");

        // comment
        // bar();
    } else if false {
        println!("bar");
    } else if true {$0
        println!("foo");
    }
}
"#,
            r#"
fn main() {
    // bar();
    if true {
        println!("true");

        // comment
        // bar();
    } else if false {
        println!("bar");
    }
    println!("foo");
}
"#,
        );
    }

    #[test]
    fn simple_if_else_if_nested_else() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    // bar();
    if true {
        println!("true");

        // comment
        // bar();
    } else if false {
        println!("bar");
    } else if true {
        println!("foo");
    } else {$0
        println!("else");
    }
}
"#,
            r#"
fn main() {
    // bar();
    if true {
        println!("true");

        // comment
        // bar();
    } else if false {
        println!("bar");
    } else if true {
        println!("foo");
    }
    println!("else");
}
"#,
        );
    }

    #[test]
    fn simple_if_else_if_nested_middle() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    // bar();
    if true {
        println!("true");

        // comment
        // bar();
    } else if false {
        println!("bar");
    } else if true {$0
        println!("foo");
    } else {
        println!("else");
    }
}
"#,
            r#"
fn main() {
    // bar();
    if true {
        println!("true");

        // comment
        // bar();
    } else if false {
        println!("bar");
    }
    println!("foo");
}
"#,
        );
    }

    #[test]
    fn simple_if_bad_cursor_position() {
        check_assist_not_applicable(
            unwrap_block,
            r#"
fn main() {
    bar();$0
    if true {
        foo();

        // comment
        bar();
    } else {
        println!("bar");
    }
}
"#,
        );
    }

    #[test]
    fn simple_for() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    for i in 0..5 {$0
        if true {
            foo();

            // comment
            bar();
        } else {
            println!("bar");
        }
    }
}
"#,
            r#"
fn main() {
    if true {
        foo();

        // comment
        bar();
    } else {
        println!("bar");
    }
}
"#,
        );
    }

    #[test]
    fn simple_if_in_for() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    for i in 0..5 {
        if true {$0
            foo();

            // comment
            bar();
        } else {
            println!("bar");
        }
    }
}
"#,
            r#"
fn main() {
    for i in 0..5 {
        foo();

        // comment
        bar();
    }
}
"#,
        );
    }

    #[test]
    fn simple_loop() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    loop {$0
        if true {
            foo();

            // comment
            bar();
        } else {
            println!("bar");
        }
    }
}
"#,
            r#"
fn main() {
    if true {
        foo();

        // comment
        bar();
    } else {
        println!("bar");
    }
}
"#,
        );
    }

    #[test]
    fn simple_while() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    while true {$0
        if true {
            foo();

            // comment
            bar();
        } else {
            println!("bar");
        }
    }
}
"#,
            r#"
fn main() {
    if true {
        foo();

        // comment
        bar();
    } else {
        println!("bar");
    }
}
"#,
        );
    }

    #[test]
    fn unwrap_match_arm() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    match rel_path {
        Ok(rel_path) => {$0
            let rel_path = RelativePathBuf::from_path(rel_path).ok()?;
            Some((*id, rel_path))
        }
        Err(_) => None,
    }
}
"#,
            r#"
fn main() {
    let rel_path = RelativePathBuf::from_path(rel_path).ok()?;
    Some((*id, rel_path))
}
"#,
        );
    }

    #[test]
    fn simple_if_in_while_bad_cursor_position() {
        check_assist_not_applicable(
            unwrap_block,
            r#"
fn main() {
    while true {
        if true {
            foo();$0

            // comment
            bar();
        } else {
            println!("bar");
        }
    }
}
"#,
        );
    }

    #[test]
    fn simple_single_line() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    {$0 0 }
}
"#,
            r#"
fn main() {
    0
}
"#,
        );
    }

    #[test]
    fn simple_nested_block() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    $0{
        {
            3
        }
    }
}
"#,
            r#"
fn main() {
    {
        3
    }
}
"#,
        );
    }

    #[test]
    fn nested_single_line() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    {$0 { println!("foo"); } }
}
"#,
            r#"
fn main() {
    { println!("foo"); }
}
"#,
        );

        check_assist(
            unwrap_block,
            r#"
fn main() {
    {$0 { 0 } }
}
"#,
            r#"
fn main() {
    { 0 }
}
"#,
        );
    }

    #[test]
    fn simple_if_single_line() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    if true {$0 /* foo */ foo() } else { bar() /* bar */}
}
"#,
            r#"
fn main() {
    /* foo */ foo()
}
"#,
        );
    }

    #[test]
    fn if_single_statement() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    if true {$0
        return 3;
    }
}
"#,
            r#"
fn main() {
    return 3;
}
"#,
        );
    }

    #[test]
    fn multiple_statements() {
        check_assist(
            unwrap_block,
            r#"
fn main() -> i32 {
    if 2 > 1 {$0
        let a = 5;
        return 3;
    }
    5
}
"#,
            r#"
fn main() -> i32 {
    let a = 5;
    return 3;
    5
}
"#,
        );
    }

    #[test]
    fn unwrap_block_in_let_initializers() {
        // https://github.com/rust-lang/rust-analyzer/issues/13679
        check_assist(
            unwrap_block,
            r#"
fn main() {
    let x = {$0};
}
"#,
            r#"
fn main() {
    let x = ();
}
"#,
        );
        check_assist(
            unwrap_block,
            r#"
fn main() {
    let x = {$0
        bar
    };
}
"#,
            r#"
fn main() {
    let x = bar;
}
"#,
        );
        check_assist(
            unwrap_block,
            r#"
fn main() -> i32 {
    let _ = {$01; 2};
}
"#,
            r#"
fn main() -> i32 {
    1;
    let _ = 2;
}
"#,
        );
        check_assist(
            unwrap_block,
            r#"
fn main() -> i32 {
    let mut a = {$01; 2};
}
"#,
            r#"
fn main() -> i32 {
    1;
    let mut a = 2;
}
"#,
        );
    }

    #[test]
    fn unwrap_if_in_let_initializers() {
        // https://github.com/rust-lang/rust-analyzer/issues/13679
        check_assist(
            unwrap_block,
            r#"
fn main() {
    let a = 1;
    let x = if a - 1 == 0 {$0
        foo
    } else {
        bar
    };
}
"#,
            r#"
fn main() {
    let a = 1;
    let x = foo;
}
"#,
        );
    }

    #[test]
    fn unwrap_block_with_modifiers() {
        // https://github.com/rust-lang/rust-analyzer/issues/17964
        check_assist(
            unwrap_block,
            r#"
fn main() {
    unsafe $0{
        bar;
    }
}
"#,
            r#"
fn main() {
    bar;
}
"#,
        );
        check_assist(
            unwrap_block,
            r#"
fn main() {
    async move $0{
        bar;
    }
}
"#,
            r#"
fn main() {
    bar;
}
"#,
        );
        check_assist(
            unwrap_block,
            r#"
fn main() {
    try $0{
        bar;
    }
}
"#,
            r#"
fn main() {
    bar;
}
"#,
        );
    }
}
