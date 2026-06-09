use either::Either;
use syntax::{
    AstNode, SyntaxElement, SyntaxKind, SyntaxNode, T,
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        syntax_factory::SyntaxFactory,
    },
    match_ast,
    syntax_editor::{Element, Position, SyntaxEditor},
};

use crate::{AssistContext, AssistId, Assists};

// Assist: unwrap_branch
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
pub(crate) fn unwrap_branch(acc: &mut Assists, ctx: &AssistContext<'_, '_>) -> Option<()> {
    let (editor, _) = SyntaxEditor::new(ctx.source_file().syntax().clone());
    let place = unwrap_branch_place(ctx)?;
    let target = place.syntax().text_range();
    let block = wrap_block_raw(&place, editor.make());
    let mut container = place.syntax().clone();
    let mut replacement = block.clone();
    let mut prefer_container = None;

    let from_indent = place.indent_level();
    let into_indent = loop {
        let parent = container.parent()?;
        container = match_ast! {
            match parent {
                ast::ForExpr(it) => it.syntax().clone(),
                ast::LoopExpr(it) => it.syntax().clone(),
                ast::WhileExpr(it) => it.syntax().clone(),
                ast::MatchArm(it) => it.parent_match().syntax().clone(),
                ast::LetElse(it) => it.syntax().parent()?,
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
        if ast::MatchArm::cast(container.parent()?).is_some() {
            replacement = editor.make().tail_only_block_expr(replacement.into());
            prefer_container = Some(container.clone());
            break IndentLevel::from_node(&container);
        }
    };
    let is_branch =
        !block.is_standalone() || place.syntax().parent().and_then(ast::MatchArm::cast).is_some();
    let label = if is_branch { "Unwrap branch" } else { "Unwrap block" };
    let replacement = replacement.stmt_list()?;

    acc.add(AssistId::refactor_rewrite("unwrap_branch"), label, target, |builder| {
        let replacement = replacement.dedent(from_indent).indent(into_indent);
        let mut replacement = extract_statements(replacement);
        let container = prefer_container.unwrap_or(container);

        if ast::ExprStmt::can_cast(container.kind())
            && block.tail_expr().is_some_and(|it| !it.is_block_like())
        {
            replacement.push(editor.make().token(T![;]).into());
        }

        editor.replace_with_many(&container, replacement);
        delete_else_before(container, &editor);

        builder.add_file_edits(ctx.vfs_file_id(), editor);
    })
}

// Assist: unwrap_block
//
// This assist removes braces and unwrap single expressions block.
//
// ```
// fn foo() {
//     match () {
//         _ => {$0
//             bar()
//         }
//     }
// }
// ```
// ->
// ```
// fn foo() {
//     match () {
//         _ => bar(),
//     }
// }
// ```
pub(crate) fn unwrap_block(acc: &mut Assists, ctx: &AssistContext<'_, '_>) -> Option<()> {
    let l_curly_token = ctx.find_token_syntax_at_offset(T!['{'])?;
    let block = l_curly_token.parent_ancestors().nth(1).and_then(ast::BlockExpr::cast)?;
    let target = block.syntax().text_range();
    let tail_expr = block.tail_expr()?;
    let stmt_list = block.stmt_list()?;
    let container = Either::<ast::MatchArm, ast::ClosureExpr>::cast(block.syntax().parent()?)?;

    if stmt_list.statements().next().is_some() {
        return None;
    }

    acc.add(AssistId::refactor_rewrite("unwrap_block"), "Unwrap block", target, |builder| {
        let editor = builder.make_editor(block.syntax());
        let replacement = stmt_list.dedent(tail_expr.indent_level()).indent(block.indent_level());
        let mut replacement = extract_statements(replacement);

        if container.left().is_some_and(|it| it.comma_token().is_none())
            && !tail_expr.is_block_like()
        {
            replacement.push(editor.make().token(T![,]).into());
        }

        editor.replace_with_many(block.syntax(), replacement);
        builder.add_file_edits(ctx.vfs_file_id(), editor);
    })
}

fn delete_else_before(container: SyntaxNode, editor: &SyntaxEditor) {
    let make = editor.make();
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
        .for_each(|it| editor.delete(it));
    let indent = IndentLevel::from_node(&container);
    let newline = make.whitespace(&format!("\n{indent}"));
    editor.replace(else_token, newline);
}

fn wrap_let(assign: &ast::LetStmt, replacement: ast::BlockExpr) -> ast::BlockExpr {
    let try_wrap_assign = || {
        let initializer = assign.initializer()?.syntax().syntax_element();
        let (editor, replacement) = SyntaxEditor::with_ast_node(&replacement);
        let tail_expr = replacement.tail_expr()?;
        let before =
            assign.syntax().children_with_tokens().take_while(|it| *it != initializer).collect();
        let after = assign
            .syntax()
            .children_with_tokens()
            .skip_while(|it| *it != initializer)
            .skip(1)
            .collect();

        editor.insert_all(Position::before(tail_expr.syntax()), before);
        editor.insert_all(Position::after(tail_expr.syntax()), after);
        ast::BlockExpr::cast(editor.finish().new_root().clone())
    };
    try_wrap_assign().unwrap_or(replacement)
}

fn unwrap_branch_place(ctx: &AssistContext<'_, '_>) -> Option<ast::Expr> {
    if let Some(l_curly_token) = ctx.find_token_syntax_at_offset(T!['{']) {
        let block = l_curly_token.parent_ancestors().nth(1).and_then(ast::BlockExpr::cast)?;
        Some(block.into())
    } else if let Some(fat_arrow_token) = ctx.find_token_syntax_at_offset(T![=>]) {
        let match_arm = fat_arrow_token.parent().and_then(ast::MatchArm::cast)?;
        match_arm.expr()
    } else {
        None
    }
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

fn wrap_block_raw(expr: &ast::Expr, make: &SyntaxFactory) -> ast::BlockExpr {
    if let ast::Expr::BlockExpr(block) = expr {
        block.clone()
    } else {
        make.tail_only_block_expr(expr.indent(1.into()))
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{
        check_assist, check_assist_by_label, check_assist_not_applicable,
        check_assist_not_applicable_by_label, check_assist_with_label,
    };

    use super::*;

    #[test]
    fn unwrap_tail_expr_block() {
        check_assist(
            unwrap_branch,
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
            unwrap_branch,
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
        check_assist(
            unwrap_branch,
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
    92;
    ()
}
"#,
        );
    }

    #[test]
    fn simple_if() {
        check_assist(
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
    fn simple_if_in_match_arm() {
        check_assist(
            unwrap_branch,
            r#"
fn main() {
    match 1 {
        1 => if true {$0
            foo();
        }
        _ => (),
    }
}
"#,
            r#"
fn main() {
    match 1 {
        1 => {
            foo();
        }
        _ => (),
    }
}
"#,
        );

        check_assist(
            unwrap_branch,
            r#"
fn main() {
    match 1 {
        1 => if true {
            foo();
        } else {$0
            bar();
        }
        _ => (),
    }
}
"#,
            r#"
fn main() {
    match 1 {
        1 => {
            bar();
        }
        _ => (),
    }
}
"#,
        );
    }

    #[test]
    fn simple_match_in_match_arm() {
        check_assist(
            unwrap_branch,
            r#"
fn main() {
    match 1 {
        1 => match () {
            _ => {$0
                foo();
            }
        }
        _ => (),
    }
}
"#,
            r#"
fn main() {
    match 1 {
        1 => {
            foo();
        }
        _ => (),
    }
}
"#,
        );
    }

    #[test]
    fn simple_loop() {
        check_assist(
            unwrap_branch,
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
            unwrap_branch,
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
    fn simple_let_else() {
        check_assist(
            unwrap_branch,
            r#"
fn main() {
    let Some(2) = None else {$0
        return;
    };
}
"#,
            r#"
fn main() {
    return;
}
"#,
        );
        check_assist(
            unwrap_branch,
            r#"
fn main() {
    let Some(2) = None else {$0
        return
    };
}
"#,
            r#"
fn main() {
    return
}
"#,
        );
    }

    #[test]
    fn unwrap_match_arm() {
        check_assist(
            unwrap_branch,
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
            unwrap_branch,
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
    fn unwrap_match_arm_without_block() {
        check_assist(
            unwrap_branch,
            r#"
fn main() {
    match rel_path {
        Ok(rel_path) $0=> Foo {
            rel_path,
        },
        Err(_) => None,
    }
}
"#,
            r#"
fn main() {
    Foo {
        rel_path,
    }
}
"#,
        );
    }

    #[test]
    fn simple_if_in_while_bad_cursor_position() {
        check_assist_not_applicable(
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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
            unwrap_branch,
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

    #[test]
    fn unwrap_block_labels() {
        check_assist_with_label(
            unwrap_branch,
            r#"
fn main() {
    $0{
        bar;
    }
}
"#,
            "Unwrap block",
        );
        check_assist_with_label(
            unwrap_branch,
            r#"
fn main() {
    let x = $0{
        bar()
    };
}
"#,
            "Unwrap block",
        );
        check_assist_with_label(
            unwrap_branch,
            r#"
fn main() {
    let x = if true $0{
        bar()
    };
}
"#,
            "Unwrap branch",
        );
        check_assist_with_label(
            unwrap_branch,
            r#"
fn main() {
    let x = match () {
        () => $0{
            bar(),
        }
    };
}
"#,
            "Unwrap branch",
        );
        check_assist_with_label(
            unwrap_branch,
            r#"
fn main() {
    match () {
        () => $0{
            bar(),
        }
    }
}
"#,
            "Unwrap branch",
        );
    }

    #[test]
    fn unwrap_block_in_branch() {
        check_assist_by_label(
            unwrap_block,
            r#"
fn main() {
    match rel_path {
        Ok(rel_path) => {$0
            if true {
                foo()
            }
        }
        Err(_) => None,
    }
}
"#,
            r#"
fn main() {
    match rel_path {
        Ok(rel_path) => if true {
            foo()
        }
        Err(_) => None,
    }
}
"#,
            "Unwrap block",
        );

        check_assist_by_label(
            unwrap_block,
            r#"
fn main() {
    match rel_path {
        Ok(rel_path) => {$0
            1 + 2
        }
        Err(_) => None,
    }
}
"#,
            r#"
fn main() {
    match rel_path {
        Ok(rel_path) => 1 + 2,
        Err(_) => None,
    }
}
"#,
            "Unwrap block",
        );
    }

    #[test]
    fn unwrap_block_in_branch_non_standalone() {
        check_assist_not_applicable_by_label(
            unwrap_block,
            r#"
fn main() {
    match rel_path {
        Ok(rel_path) => {
            if true {$0
                foo()
            }
        }
        Err(_) => None,
    }
}
"#,
            "Unwrap block",
        );
    }

    #[test]
    fn unwrap_block_in_branch_non_tail_expr_only() {
        check_assist_not_applicable_by_label(
            unwrap_block,
            r#"
fn main() {
    match rel_path {
        Ok(rel_path) => {$0
            x;
            y
        }
        Err(_) => None,
    }
}
"#,
            "Unwrap block",
        );
    }

    #[test]
    fn unwrap_block_in_closure() {
        check_assist_by_label(
            unwrap_block,
            r#"
fn main() {
    let f = || {$0 foo() };
}
"#,
            r#"
fn main() {
    let f = || foo();
}
"#,
            "Unwrap block",
        );
    }
}
