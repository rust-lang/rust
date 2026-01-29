use syntax::{
    AstNode, SyntaxElement, SyntaxKind, SyntaxNode, T,
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make,
    },
    match_ast,
    syntax_editor::{Element, Position, SyntaxEditor},
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
    let l_curly_token = ctx.find_token_syntax_at_offset(T!['{'])?;
    let block = l_curly_token.parent_ancestors().nth(1).and_then(ast::BlockExpr::cast)?;
    let target = block.syntax().text_range();
    let mut container = block.syntax().clone();
    let mut replacement = block.clone();
    let mut prefer_container = None;

    let from_indent = block.indent_level();
    let into_indent = loop {
        let parent = container.parent()?;
        container = match_ast! {
            match parent {
                ast::ForExpr(it) => it.syntax().clone(),
                ast::LoopExpr(it) => it.syntax().clone(),
                ast::WhileExpr(it) => it.syntax().clone(),
                ast::MatchArm(it) => it.parent_match().syntax().clone(),
                ast::LetStmt(it) => {
                    replacement = wrap_let(&it, replacement);
                    prefer_container = Some(it.syntax().clone());
                    it.syntax().clone()
                },
                ast::IfExpr(it) => {
                    prefer_container.get_or_insert_with(|| {
                        if let Some(else_branch) = it.else_branch()
                            && *else_branch.syntax() == container
                        {
                            else_branch.syntax().clone()
                        } else {
                            it.syntax().clone()
                        }
                    });
                    it.syntax().clone()
                },
                ast::ExprStmt(it) => it.syntax().clone(),
                ast::StmtList(it) => break it.indent_level(),
                _ => return None,
            }
        };
    };
    let replacement = replacement.stmt_list()?;

    acc.add(AssistId::refactor_rewrite("unwrap_block"), "Unwrap block", target, |builder| {
        let mut edit = builder.make_editor(block.syntax());
        let replacement = replacement.dedent(from_indent).indent(into_indent);
        let container = prefer_container.unwrap_or(container);

        edit.replace_with_many(&container, extract_statements(replacement));
        delete_else_before(container, &mut edit);

        builder.add_file_edits(ctx.vfs_file_id(), edit);
    })
}

fn delete_else_before(container: SyntaxNode, edit: &mut SyntaxEditor) {
    let Some(else_token) = container
        .siblings_with_tokens(syntax::Direction::Prev)
        .skip(1)
        .map_while(|it| it.into_token())
        .find(|it| it.kind() == T![else])
    else {
        return;
    };
    itertools::chain(else_token.prev_token(), else_token.next_token())
        .filter(|it| it.kind() == SyntaxKind::WHITESPACE)
        .for_each(|it| edit.delete(it));
    let indent = IndentLevel::from_node(&container);
    let newline = make::tokens::whitespace(&format!("\n{indent}"));
    edit.replace(else_token, newline);
}

fn wrap_let(assign: &ast::LetStmt, replacement: ast::BlockExpr) -> ast::BlockExpr {
    let try_wrap_assign = || {
        let initializer = assign.initializer()?.syntax().syntax_element();
        let replacement = replacement.clone_subtree();
        let assign = assign.clone_for_update();
        let tail_expr = replacement.tail_expr()?;
        let before =
            assign.syntax().children_with_tokens().take_while(|it| *it != initializer).collect();
        let after = assign
            .syntax()
            .children_with_tokens()
            .skip_while(|it| *it != initializer)
            .skip(1)
            .collect();

        let mut edit = SyntaxEditor::new(replacement.syntax().clone());
        edit.insert_all(Position::before(tail_expr.syntax()), before);
        edit.insert_all(Position::after(tail_expr.syntax()), after);
        ast::BlockExpr::cast(edit.finish().new_root().clone())
    };
    try_wrap_assign().unwrap_or(replacement)
}

fn extract_statements(stmt_list: ast::StmtList) -> Vec<SyntaxElement> {
    let mut elements = stmt_list
        .syntax()
        .children_with_tokens()
        .filter(|it| !matches!(it.kind(), T!['{'] | T!['}']))
        .skip_while(|it| it.kind() == SyntaxKind::WHITESPACE)
        .collect::<Vec<_>>();
    while elements.pop_if(|it| it.kind() == SyntaxKind::WHITESPACE).is_some() {}
    elements
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
    fn unwrap_match_arm_in_let() {
        check_assist(
            unwrap_block,
            r#"
fn main() {
    let value = match rel_path {
        Ok(rel_path) => {$0
            let rel_path = RelativePathBuf::from_path(rel_path).ok()?;
            Some((*id, rel_path))
        }
        Err(_) => None,
    };
}
"#,
            r#"
fn main() {
    let rel_path = RelativePathBuf::from_path(rel_path).ok()?;
    let value = Some((*id, rel_path));
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
    1; let _ = 2;
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
    1; let mut a = 2;
}
"#,
        );
        check_assist(
            unwrap_block,
            r#"
fn main() -> i32 {
    let mut a = {$0
        1;
        2;
        3
    };
}
"#,
            r#"
fn main() -> i32 {
    1;
    2;
    let mut a = 3;
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
