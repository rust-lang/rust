use hir::TypeInfo;
use ide_db::syntax_helpers::suggest_name;
use syntax::{
    ast::{
        self, edit::IndentLevel, edit_in_place::Indent, make, syntax_factory::SyntaxFactory,
        AstNode,
    },
    syntax_editor::Position,
    NodeOrToken,
    SyntaxKind::{BLOCK_EXPR, BREAK_EXPR, COMMENT, LOOP_EXPR, MATCH_GUARD, PATH_EXPR, RETURN_EXPR},
    SyntaxNode, T,
};

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
//     let $0var_name = 1 + 2;
//     var_name * 4;
// }
// ```
pub(crate) fn extract_variable(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let node = if ctx.has_empty_selection() {
        if let Some(t) = ctx.token_at_offset().find(|it| it.kind() == T![;]) {
            t.parent().and_then(ast::ExprStmt::cast)?.syntax().clone()
        } else if let Some(expr) = ctx.find_node_at_offset::<ast::Expr>() {
            expr.syntax().ancestors().find_map(valid_target_expr)?.syntax().clone()
        } else {
            return None;
        }
    } else {
        match ctx.covering_element() {
            NodeOrToken::Node(it) => it,
            NodeOrToken::Token(it) if it.kind() == COMMENT => {
                cov_mark::hit!(extract_var_in_comment_is_not_applicable);
                return None;
            }
            NodeOrToken::Token(it) => it.parent()?,
        }
    };

    let node = node.ancestors().take_while(|anc| anc.text_range() == node.text_range()).last()?;
    let range = node.text_range();

    let to_extract = node
        .descendants()
        .take_while(|it| range.contains_range(it.text_range()))
        .find_map(valid_target_expr)?;

    let ty = ctx.sema.type_of_expr(&to_extract).map(TypeInfo::adjusted);
    if matches!(&ty, Some(ty_info) if ty_info.is_unit()) {
        return None;
    }

    let parent = to_extract.syntax().parent().and_then(ast::Expr::cast);
    // Any expression that autoderefs may need adjustment.
    let mut needs_adjust = parent.as_ref().map_or(false, |it| match it {
        ast::Expr::FieldExpr(_)
        | ast::Expr::MethodCallExpr(_)
        | ast::Expr::CallExpr(_)
        | ast::Expr::AwaitExpr(_) => true,
        ast::Expr::IndexExpr(index) if index.base().as_ref() == Some(&to_extract) => true,
        _ => false,
    });
    let mut to_extract_no_ref = peel_parens(to_extract.clone());
    let needs_ref = needs_adjust
        && match &to_extract_no_ref {
            ast::Expr::FieldExpr(_)
            | ast::Expr::IndexExpr(_)
            | ast::Expr::MacroExpr(_)
            | ast::Expr::ParenExpr(_)
            | ast::Expr::PathExpr(_) => true,
            ast::Expr::PrefixExpr(prefix) if prefix.op_kind() == Some(ast::UnaryOp::Deref) => {
                to_extract_no_ref = prefix.expr()?;
                needs_adjust = false;
                false
            }
            _ => false,
        };

    let anchor = Anchor::from(&to_extract)?;
    let target = to_extract.syntax().text_range();
    acc.add(
        AssistId("extract_variable", AssistKind::RefactorExtract),
        "Extract into variable",
        target,
        move |edit| {
            let field_shorthand = to_extract
                .syntax()
                .parent()
                .and_then(ast::RecordExprField::cast)
                .filter(|field| field.name_ref().is_some());

            let (var_name, expr_replace) = match field_shorthand {
                Some(field) => (field.to_string(), field.syntax().clone()),
                None => (
                    suggest_name::for_variable(&to_extract, &ctx.sema),
                    to_extract.syntax().clone(),
                ),
            };

            let make = SyntaxFactory::new();
            let mut editor = edit.make_editor(&expr_replace);

            let pat_name = make.name(&var_name);
            let name_expr = make.expr_path(make::ext::ident_path(&var_name));

            if let Some(cap) = ctx.config.snippet_cap {
                let tabstop = edit.make_tabstop_before(cap);
                editor.add_annotation(pat_name.syntax().clone(), tabstop);
            }

            let ident_pat = match parent {
                Some(ast::Expr::RefExpr(expr)) if expr.mut_token().is_some() => {
                    make.ident_pat(false, true, pat_name)
                }
                _ if needs_adjust
                    && !needs_ref
                    && ty.as_ref().is_some_and(|ty| ty.is_mutable_reference()) =>
                {
                    make.ident_pat(false, true, pat_name)
                }
                _ => make.ident_pat(false, false, pat_name),
            };

            let to_extract_no_ref = match ty.as_ref().filter(|_| needs_ref) {
                Some(receiver_type) if receiver_type.is_mutable_reference() => {
                    make.expr_ref(to_extract_no_ref, true)
                }
                Some(receiver_type) if receiver_type.is_reference() => {
                    make.expr_ref(to_extract_no_ref, false)
                }
                _ => to_extract_no_ref,
            };

            let let_stmt = make.let_stmt(ident_pat.into(), None, Some(to_extract_no_ref));

            match anchor {
                Anchor::Before(place) => {
                    let prev_ws = place.prev_sibling_or_token().and_then(|it| it.into_token());
                    let indent_to = IndentLevel::from_node(&place);

                    // Adjust ws to insert depending on if this is all inline or on separate lines
                    let trailing_ws = if prev_ws.is_some_and(|it| it.text().starts_with('\n')) {
                        format!("\n{indent_to}")
                    } else {
                        " ".to_owned()
                    };

                    editor.insert_all(
                        Position::before(place),
                        vec![
                            let_stmt.syntax().clone().into(),
                            make::tokens::whitespace(&trailing_ws).into(),
                        ],
                    );

                    editor.replace(expr_replace, name_expr.syntax());
                }
                Anchor::Replace(stmt) => {
                    cov_mark::hit!(test_extract_var_expr_stmt);

                    editor.replace(stmt.syntax(), let_stmt.syntax());
                }
                Anchor::WrapInBlock(to_wrap) => {
                    let indent_to = to_wrap.indent_level();

                    let block = if to_wrap.syntax() == &expr_replace {
                        // Since `expr_replace` is the same that needs to be wrapped in a block,
                        // we can just directly replace it with a block
                        make.block_expr([let_stmt.into()], Some(name_expr))
                    } else {
                        // `expr_replace` is a descendant of `to_wrap`, so we just replace it with `name_expr`.
                        editor.replace(expr_replace, name_expr.syntax());
                        make.block_expr([let_stmt.into()], Some(to_wrap.clone()))
                    };

                    editor.replace(to_wrap.syntax(), block.syntax());

                    // fixup indentation of block
                    block.indent(indent_to);
                }
            }

            editor.add_mappings(make.finish_with_mappings());
            edit.add_file_edits(ctx.file_id(), editor);
            edit.rename();
        },
    )
}

fn peel_parens(mut expr: ast::Expr) -> ast::Expr {
    while let ast::Expr::ParenExpr(parens) = &expr {
        let Some(expr_inside) = parens.expr() else { break };
        expr = expr_inside;
    }
    expr
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

#[derive(Debug)]
enum Anchor {
    Before(SyntaxNode),
    Replace(ast::ExprStmt),
    WrapInBlock(ast::Expr),
}

impl Anchor {
    fn from(to_extract: &ast::Expr) -> Option<Anchor> {
        to_extract
            .syntax()
            .ancestors()
            .take_while(|it| !ast::Item::can_cast(it.kind()) || ast::MacroCall::can_cast(it.kind()))
            .find_map(|node| {
                if ast::MacroCall::can_cast(node.kind()) {
                    return None;
                }
                if let Some(expr) =
                    node.parent().and_then(ast::StmtList::cast).and_then(|it| it.tail_expr())
                {
                    if expr.syntax() == &node {
                        cov_mark::hit!(test_extract_var_last_expr);
                        return Some(Anchor::Before(node));
                    }
                }

                if let Some(parent) = node.parent() {
                    if let Some(parent) = ast::ClosureExpr::cast(parent.clone()) {
                        cov_mark::hit!(test_extract_var_in_closure_no_block);
                        return parent.body().map(Anchor::WrapInBlock);
                    }
                    if let Some(parent) = ast::MatchArm::cast(parent) {
                        if node.kind() == MATCH_GUARD {
                            cov_mark::hit!(test_extract_var_in_match_guard);
                        } else {
                            cov_mark::hit!(test_extract_var_in_match_arm_no_block);
                            return parent.expr().map(Anchor::WrapInBlock);
                        }
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
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

    #[test]
    fn test_extract_var_simple_without_select() {
        check_assist(
            extract_variable,
            r#"
fn main() -> i32 {
    if true {
        1
    } else {
        2
    }$0
}
"#,
            r#"
fn main() -> i32 {
    let $0var_name = if true {
        1
    } else {
        2
    };
    var_name
}
"#,
        );

        check_assist(
            extract_variable,
            r#"
fn foo() -> i32 { 1 }
fn main() {
    foo();$0
}
"#,
            r#"
fn foo() -> i32 { 1 }
fn main() {
    let $0foo = foo();
}
"#,
        );

        check_assist(
            extract_variable,
            r#"
fn main() {
    let a = Some(2);
    a.is_some();$0
}
"#,
            r#"
fn main() {
    let a = Some(2);
    let $0is_some = a.is_some();
}
"#,
        );

        check_assist(
            extract_variable,
            r#"
fn main() {
    "hello"$0;
}
"#,
            r#"
fn main() {
    let $0var_name = "hello";
}
"#,
        );

        check_assist(
            extract_variable,
            r#"
fn main() {
    1  + 2$0;
}
"#,
            r#"
fn main() {
    let $0var_name = 1  + 2;
}
"#,
        );

        check_assist(
            extract_variable,
            r#"
fn main() {
    match () {
        () if true => 1,
        _ => 2,
    };$0
}
"#,
            r#"
fn main() {
    let $0var_name = match () {
        () if true => 1,
        _ => 2,
    };
}
"#,
        );
    }

    #[test]
    fn test_extract_var_unit_expr_without_select_not_applicable() {
        check_assist_not_applicable(
            extract_variable,
            r#"
fn foo() {}
fn main() {
    foo()$0;
}
"#,
        );

        check_assist_not_applicable(
            extract_variable,
            r#"
fn foo() {
    let mut i = 3;
    if i >= 0 {
        i += 1;
    } else {
        i -= 1;
    }$0
}"#,
        );
    }

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
        cov_mark::check!(extract_var_in_comment_is_not_applicable);
        check_assist_not_applicable(extract_variable, "fn main() { 1 + /* $0comment$0 */ 1; }");
    }

    #[test]
    fn test_extract_var_expr_stmt() {
        cov_mark::check!(test_extract_var_expr_stmt);
        check_assist(
            extract_variable,
            r#"
fn foo() {
  $0  1 + 1$0;
}"#,
            r#"
fn foo() {
    let $0var_name = 1 + 1;
}"#,
        );
        check_assist(
            extract_variable,
            r"
fn foo() {
    $0{ let x = 0; x }$0;
    something_else();
}",
            r"
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
            r"
fn foo() {
    $01$0 + 1;
}",
            r"
fn foo() {
    let $0var_name = 1;
    var_name + 1;
}",
        );
    }

    #[test]
    fn test_extract_var_last_expr() {
        cov_mark::check!(test_extract_var_last_expr);
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
fn foo() -> i32 {
    $0bar(1 + 1)$0
}

fn bar(i: i32) -> i32 {
    i
}
"#,
            r#"
fn foo() -> i32 {
    let $0bar = bar(1 + 1);
    bar
}

fn bar(i: i32) -> i32 {
    i
}
"#,
        )
    }

    #[test]
    fn test_extract_var_in_match_arm_no_block() {
        cov_mark::check!(test_extract_var_in_match_arm_no_block);
        check_assist(
            extract_variable,
            r#"
fn main() {
    let x = true;
    let tuple = match x {
        true => ($02 + 2$0, true)
        _ => (0, false)
    };
}
"#,
            r#"
fn main() {
    let x = true;
    let tuple = match x {
        true => {
            let $0var_name = 2 + 2;
            (var_name, true)
        }
        _ => (0, false)
    };
}
"#,
        );
    }

    #[test]
    fn test_extract_var_in_match_arm_with_block() {
        check_assist(
            extract_variable,
            r#"
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
"#,
            r#"
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
"#,
        );
    }

    #[test]
    fn test_extract_var_in_match_guard() {
        cov_mark::check!(test_extract_var_in_match_guard);
        check_assist(
            extract_variable,
            r#"
fn main() {
    match () {
        () if $010 > 0$0 => 1
        _ => 2
    };
}
"#,
            r#"
fn main() {
    let $0var_name = 10 > 0;
    match () {
        () if var_name => 1
        _ => 2
    };
}
"#,
        );
    }

    #[test]
    fn test_extract_var_in_closure_no_block() {
        cov_mark::check!(test_extract_var_in_closure_no_block);
        check_assist(
            extract_variable,
            r#"
fn main() {
    let lambda = |x: u32| $0x * 2$0;
}
"#,
            r#"
fn main() {
    let lambda = |x: u32| {
        let $0var_name = x * 2;
        var_name
    };
}
"#,
        );
    }

    #[test]
    fn test_extract_var_in_closure_with_block() {
        check_assist(
            extract_variable,
            r#"
fn main() {
    let lambda = |x: u32| { $0x * 2$0 };
}
"#,
            r#"
fn main() {
    let lambda = |x: u32| { let $0var_name = x * 2; var_name };
}
"#,
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
    let $0foo = bar.foo();
    let v = foo;
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
    fn extract_var_name_from_type() {
        check_assist(
            extract_variable,
            r#"
struct Test(i32);

fn foo() -> Test {
    $0{ Test(10) }$0
}
"#,
            r#"
struct Test(i32);

fn foo() -> Test {
    let $0test = { Test(10) };
    test
}
"#,
        )
    }

    #[test]
    fn extract_var_name_from_parameter() {
        check_assist(
            extract_variable,
            r#"
fn bar(test: u32, size: u32)

fn foo() {
    bar(1, $01+1$0);
}
"#,
            r#"
fn bar(test: u32, size: u32)

fn foo() {
    let $0size = 1+1;
    bar(1, size);
}
"#,
        )
    }

    #[test]
    fn extract_var_parameter_name_has_precedence_over_type() {
        check_assist(
            extract_variable,
            r#"
struct TextSize(u32);
fn bar(test: u32, size: TextSize)

fn foo() {
    bar(1, $0{ TextSize(1+1) }$0);
}
"#,
            r#"
struct TextSize(u32);
fn bar(test: u32, size: TextSize)

fn foo() {
    let $0size = { TextSize(1+1) };
    bar(1, size);
}
"#,
        )
    }

    #[test]
    fn extract_var_name_from_function() {
        check_assist(
            extract_variable,
            r#"
fn is_required(test: u32, size: u32) -> bool

fn foo() -> bool {
    $0is_required(1, 2)$0
}
"#,
            r#"
fn is_required(test: u32, size: u32) -> bool

fn foo() -> bool {
    let $0is_required = is_required(1, 2);
    is_required
}
"#,
        )
    }

    #[test]
    fn extract_var_name_from_method() {
        check_assist(
            extract_variable,
            r#"
struct S;
impl S {
    fn bar(&self, n: u32) -> u32 { n }
}

fn foo() -> u32 {
    $0S.bar(1)$0
}
"#,
            r#"
struct S;
impl S {
    fn bar(&self, n: u32) -> u32 { n }
}

fn foo() -> u32 {
    let $0bar = S.bar(1);
    bar
}
"#,
        )
    }

    #[test]
    fn extract_var_name_from_method_param() {
        check_assist(
            extract_variable,
            r#"
struct S;
impl S {
    fn bar(&self, n: u32, size: u32) { n }
}

fn foo() {
    S.bar($01 + 1$0, 2)
}
"#,
            r#"
struct S;
impl S {
    fn bar(&self, n: u32, size: u32) { n }
}

fn foo() {
    let $0n = 1 + 1;
    S.bar(n, 2)
}
"#,
        )
    }

    #[test]
    fn extract_var_name_from_ufcs_method_param() {
        check_assist(
            extract_variable,
            r#"
struct S;
impl S {
    fn bar(&self, n: u32, size: u32) { n }
}

fn foo() {
    S::bar(&S, $01 + 1$0, 2)
}
"#,
            r#"
struct S;
impl S {
    fn bar(&self, n: u32, size: u32) { n }
}

fn foo() {
    let $0n = 1 + 1;
    S::bar(&S, n, 2)
}
"#,
        )
    }

    #[test]
    fn extract_var_parameter_name_has_precedence_over_function() {
        check_assist(
            extract_variable,
            r#"
fn bar(test: u32, size: u32)

fn foo() {
    bar(1, $0symbol_size(1, 2)$0);
}
"#,
            r#"
fn bar(test: u32, size: u32)

fn foo() {
    let $0size = symbol_size(1, 2);
    bar(1, size);
}
"#,
        )
    }

    #[test]
    fn extract_macro_call() {
        check_assist(
            extract_variable,
            r"
struct Vec;
macro_rules! vec {
    () => {Vec}
}
fn main() {
    let _ = $0vec![]$0;
}
",
            r"
struct Vec;
macro_rules! vec {
    () => {Vec}
}
fn main() {
    let $0vec = vec![];
    let _ = vec;
}
",
        );
    }

    #[test]
    fn test_extract_var_for_return_not_applicable() {
        check_assist_not_applicable(extract_variable, "fn foo() { $0return$0; } ");
    }

    #[test]
    fn test_extract_var_for_break_not_applicable() {
        check_assist_not_applicable(extract_variable, "fn main() { loop { $0break$0; }; }");
    }

    #[test]
    fn test_extract_var_unit_expr_not_applicable() {
        check_assist_not_applicable(
            extract_variable,
            r#"
fn foo() {
    let mut i = 3;
    $0if i >= 0 {
        i += 1;
    } else {
        i -= 1;
    }$0
}"#,
        );
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

    #[test]
    fn extract_var_no_block_body() {
        check_assist_not_applicable(
            extract_variable,
            r"
const X: usize = $0100$0;
",
        );
    }

    #[test]
    fn test_extract_var_mutable_reference_parameter() {
        check_assist(
            extract_variable,
            r#"
struct S {
    vec: Vec<u8>
}

struct Vec<T>;
impl<T> Vec<T> {
    fn push(&mut self, _:usize) {}
}

fn foo(s: &mut S) {
    $0s.vec$0.push(0);
}"#,
            r#"
struct S {
    vec: Vec<u8>
}

struct Vec<T>;
impl<T> Vec<T> {
    fn push(&mut self, _:usize) {}
}

fn foo(s: &mut S) {
    let $0vec = &mut s.vec;
    vec.push(0);
}"#,
        );
    }

    #[test]
    fn test_extract_var_mutable_reference_parameter_deep_nesting() {
        check_assist(
            extract_variable,
            r#"
struct Y {
    field: X
}
struct X {
    field: S
}
struct S {
    vec: Vec<u8>
}
struct Vec<T>;
impl<T> Vec<T> {
    fn push(&mut self, _:usize) {}
}

fn foo(f: &mut Y) {
    $0f.field.field.vec$0.push(0);
}"#,
            r#"
struct Y {
    field: X
}
struct X {
    field: S
}
struct S {
    vec: Vec<u8>
}
struct Vec<T>;
impl<T> Vec<T> {
    fn push(&mut self, _:usize) {}
}

fn foo(f: &mut Y) {
    let $0vec = &mut f.field.field.vec;
    vec.push(0);
}"#,
        );
    }

    #[test]
    fn test_extract_var_reference_parameter() {
        check_assist(
            extract_variable,
            r#"
struct X;

impl X {
    fn do_thing(&self) {

    }
}

struct S {
    sub: X
}

fn foo(s: &S) {
    $0s.sub$0.do_thing();
}"#,
            r#"
struct X;

impl X {
    fn do_thing(&self) {

    }
}

struct S {
    sub: X
}

fn foo(s: &S) {
    let $0x = &s.sub;
    x.do_thing();
}"#,
        );
    }

    #[test]
    fn test_extract_var_index_deref() {
        check_assist(
            extract_variable,
            r#"
//- minicore: index
struct X;

impl std::ops::Index<usize> for X {
    type Output = i32;
    fn index(&self) -> &Self::Output { 0 }
}

struct S {
    sub: X
}

fn foo(s: &S) {
    $0s.sub$0[0];
}"#,
            r#"
struct X;

impl std::ops::Index<usize> for X {
    type Output = i32;
    fn index(&self) -> &Self::Output { 0 }
}

struct S {
    sub: X
}

fn foo(s: &S) {
    let $0sub = &s.sub;
    sub[0];
}"#,
        );
    }

    #[test]
    fn test_extract_var_reference_parameter_deep_nesting() {
        check_assist(
            extract_variable,
            r#"
struct Z;
impl Z {
    fn do_thing(&self) {

    }
}

struct Y {
    field: Z
}

struct X {
    field: Y
}

struct S {
    sub: X
}

fn foo(s: &S) {
    $0s.sub.field.field$0.do_thing();
}"#,
            r#"
struct Z;
impl Z {
    fn do_thing(&self) {

    }
}

struct Y {
    field: Z
}

struct X {
    field: Y
}

struct S {
    sub: X
}

fn foo(s: &S) {
    let $0z = &s.sub.field.field;
    z.do_thing();
}"#,
        );
    }

    #[test]
    fn test_extract_var_regular_parameter() {
        check_assist(
            extract_variable,
            r#"
struct X;

impl X {
    fn do_thing(&self) {

    }
}

struct S {
    sub: X
}

fn foo(s: S) {
    $0s.sub$0.do_thing();
}"#,
            r#"
struct X;

impl X {
    fn do_thing(&self) {

    }
}

struct S {
    sub: X
}

fn foo(s: S) {
    let $0x = &s.sub;
    x.do_thing();
}"#,
        );
    }

    #[test]
    fn test_extract_var_mutable_reference_local() {
        check_assist(
            extract_variable,
            r#"
struct X;

struct S {
    sub: X
}

impl S {
    fn new() -> S {
        S {
            sub: X::new()
        }
    }
}

impl X {
    fn new() -> X {
        X { }
    }
    fn do_thing(&self) {

    }
}


fn foo() {
    let local = &mut S::new();
    $0local.sub$0.do_thing();
}"#,
            r#"
struct X;

struct S {
    sub: X
}

impl S {
    fn new() -> S {
        S {
            sub: X::new()
        }
    }
}

impl X {
    fn new() -> X {
        X { }
    }
    fn do_thing(&self) {

    }
}


fn foo() {
    let local = &mut S::new();
    let $0x = &local.sub;
    x.do_thing();
}"#,
        );
    }

    #[test]
    fn test_extract_var_reference_local() {
        check_assist(
            extract_variable,
            r#"
struct X;

struct S {
    sub: X
}

impl S {
    fn new() -> S {
        S {
            sub: X::new()
        }
    }
}

impl X {
    fn new() -> X {
        X { }
    }
    fn do_thing(&self) {

    }
}


fn foo() {
    let local = &S::new();
    $0local.sub$0.do_thing();
}"#,
            r#"
struct X;

struct S {
    sub: X
}

impl S {
    fn new() -> S {
        S {
            sub: X::new()
        }
    }
}

impl X {
    fn new() -> X {
        X { }
    }
    fn do_thing(&self) {

    }
}


fn foo() {
    let local = &S::new();
    let $0x = &local.sub;
    x.do_thing();
}"#,
        );
    }

    #[test]
    fn test_extract_var_for_mutable_borrow() {
        check_assist(
            extract_variable,
            r#"
fn foo() {
    let v = &mut $00$0;
}"#,
            r#"
fn foo() {
    let mut $0var_name = 0;
    let v = &mut var_name;
}"#,
        );
    }

    #[test]
    fn generates_no_ref_on_calls() {
        check_assist(
            extract_variable,
            r#"
struct S;
impl S {
    fn do_work(&mut self) {}
}
fn bar() -> S { S }
fn foo() {
    $0bar()$0.do_work();
}"#,
            r#"
struct S;
impl S {
    fn do_work(&mut self) {}
}
fn bar() -> S { S }
fn foo() {
    let mut $0bar = bar();
    bar.do_work();
}"#,
        );
    }

    #[test]
    fn generates_no_ref_for_deref() {
        check_assist(
            extract_variable,
            r#"
struct S;
impl S {
    fn do_work(&mut self) {}
}
fn bar() -> S { S }
fn foo() {
    let v = &mut &mut bar();
    $0(**v)$0.do_work();
}
"#,
            r#"
struct S;
impl S {
    fn do_work(&mut self) {}
}
fn bar() -> S { S }
fn foo() {
    let v = &mut &mut bar();
    let $0s = *v;
    s.do_work();
}
"#,
        );
    }
}
