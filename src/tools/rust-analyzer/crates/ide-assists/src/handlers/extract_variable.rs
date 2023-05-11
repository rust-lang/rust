use stdx::format_to;
use syntax::{
    ast::{self, AstNode},
    NodeOrToken,
    SyntaxKind::{
        BLOCK_EXPR, BREAK_EXPR, CLOSURE_EXPR, COMMENT, LOOP_EXPR, MATCH_ARM, MATCH_GUARD,
        PATH_EXPR, RETURN_EXPR,
    },
    SyntaxNode,
};

use crate::{utils::suggest_name, AssistContext, AssistId, AssistKind, Assists};

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
pub(crate) fn extract_variable(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    if ctx.has_empty_selection() {
        return None;
    }

    let node = match ctx.covering_element() {
        NodeOrToken::Node(it) => it,
        NodeOrToken::Token(it) if it.kind() == COMMENT => {
            cov_mark::hit!(extract_var_in_comment_is_not_applicable);
            return None;
        }
        NodeOrToken::Token(it) => it.parent()?,
    };
    let node = node.ancestors().take_while(|anc| anc.text_range() == node.text_range()).last()?;
    let to_extract = node
        .descendants()
        .take_while(|it| ctx.selection_trimmed().contains_range(it.text_range()))
        .find_map(valid_target_expr)?;

    if let Some(ty_info) = ctx.sema.type_of_expr(&to_extract) {
        if ty_info.adjusted().is_unit() {
            return None;
        }
    }

    let reference_modifier = match get_receiver_type(ctx, &to_extract) {
        Some(receiver_type) if receiver_type.is_mutable_reference() => "&mut ",
        Some(receiver_type) if receiver_type.is_reference() => "&",
        _ => "",
    };

    let parent_ref_expr = to_extract.syntax().parent().and_then(ast::RefExpr::cast);
    let var_modifier = match parent_ref_expr {
        Some(expr) if expr.mut_token().is_some() => "mut ",
        _ => "",
    };

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
                None => suggest_name::for_variable(&to_extract, &ctx.sema),
            };
            let expr_range = match &field_shorthand {
                Some(it) => it.syntax().text_range().cover(to_extract.syntax().text_range()),
                None => to_extract.syntax().text_range(),
            };

            match anchor {
                Anchor::Before(_) | Anchor::Replace(_) => {
                    format_to!(buf, "let {var_modifier}{var_name} = {reference_modifier}")
                }
                Anchor::WrapInBlock(_) => {
                    format_to!(buf, "{{ let {var_name} = {reference_modifier}")
                }
            };
            format_to!(buf, "{to_extract}");

            if let Anchor::Replace(stmt) = anchor {
                cov_mark::hit!(test_extract_var_expr_stmt);
                if stmt.semicolon_token().is_none() {
                    buf.push(';');
                }
                match ctx.config.snippet_cap {
                    Some(cap) => {
                        let snip = buf.replace(
                            &format!("let {var_modifier}{var_name}"),
                            &format!("let {var_modifier}$0{var_name}"),
                        );
                        edit.replace_snippet(cap, expr_range, snip)
                    }
                    None => edit.replace(expr_range, buf),
                }
                return;
            }

            buf.push(';');

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
                    let snip = buf.replace(
                        &format!("let {var_modifier}{var_name}"),
                        &format!("let {var_modifier}$0{var_name}"),
                    );
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

fn get_receiver_type(ctx: &AssistContext<'_>, expression: &ast::Expr) -> Option<hir::Type> {
    let receiver = get_receiver(expression.clone())?;
    Some(ctx.sema.type_of_expr(&receiver)?.original())
}

/// In the expression `a.b.c.x()`, find `a`
fn get_receiver(expression: ast::Expr) -> Option<ast::Expr> {
    match expression {
        ast::Expr::FieldExpr(field) if field.expr().is_some() => {
            let nested_expression = &field.expr()?;
            get_receiver(nested_expression.to_owned())
        }
        _ => Some(expression),
    }
}

#[derive(Debug)]
enum Anchor {
    Before(SyntaxNode),
    Replace(ast::ExprStmt),
    WrapInBlock(SyntaxNode),
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
                    if parent.kind() == CLOSURE_EXPR {
                        cov_mark::hit!(test_extract_var_in_closure_no_block);
                        return Some(Anchor::WrapInBlock(node));
                    }
                    if parent.kind() == MATCH_ARM {
                        if node.kind() == MATCH_GUARD {
                            cov_mark::hit!(test_extract_var_in_match_guard);
                        } else {
                            cov_mark::hit!(test_extract_var_in_match_arm_no_block);
                            return Some(Anchor::WrapInBlock(node));
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

    fn syntax(&self) -> &SyntaxNode {
        match self {
            Anchor::Before(it) | Anchor::WrapInBlock(it) => it,
            Anchor::Replace(stmt) => stmt.syntax(),
        }
    }
}

#[cfg(test)]
mod tests {
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
        true => { let $0var_name = 2 + 2; (var_name, true) }
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
    let lambda = |x: u32| { let $0var_name = x * 2; var_name };
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

fn foo(s: &mut S) {
    $0s.vec$0.push(0);
}"#,
            r#"
struct S {
    vec: Vec<u8>
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
    let $0x = s.sub;
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
    let $0x = &mut local.sub;
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
}
