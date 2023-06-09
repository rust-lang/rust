use hir::{PathResolution, Semantics};
use ide_db::{
    base_db::FileId,
    defs::Definition,
    search::{FileReference, UsageSearchResult},
    RootDatabase,
};
use syntax::{
    ast::{self, AstNode, AstToken, HasName},
    SyntaxElement, TextRange,
};

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: inline_local_variable
//
// Inlines a local variable.
//
// ```
// fn main() {
//     let x$0 = 1 + 2;
//     x * 4;
// }
// ```
// ->
// ```
// fn main() {
//     (1 + 2) * 4;
// }
// ```
pub(crate) fn inline_local_variable(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let file_id = ctx.file_id();
    let range = ctx.selection_trimmed();
    let InlineData { let_stmt, delete_let, references, target } =
        if let Some(path_expr) = ctx.find_node_at_offset::<ast::PathExpr>() {
            inline_usage(&ctx.sema, path_expr, range, file_id)
        } else if let Some(let_stmt) = ctx.find_node_at_offset::<ast::LetStmt>() {
            inline_let(&ctx.sema, let_stmt, range, file_id)
        } else {
            None
        }?;
    let initializer_expr = let_stmt.initializer()?;

    let delete_range = delete_let.then(|| {
        if let Some(whitespace) = let_stmt
            .syntax()
            .next_sibling_or_token()
            .and_then(SyntaxElement::into_token)
            .and_then(ast::Whitespace::cast)
        {
            TextRange::new(
                let_stmt.syntax().text_range().start(),
                whitespace.syntax().text_range().end(),
            )
        } else {
            let_stmt.syntax().text_range()
        }
    });

    let wrap_in_parens = references
        .into_iter()
        .filter_map(|FileReference { range, name, .. }| match name {
            ast::NameLike::NameRef(name) => Some((range, name)),
            _ => None,
        })
        .map(|(range, name_ref)| {
            if range != name_ref.syntax().text_range() {
                // Do not rename inside macros
                // FIXME: This feels like a bad heuristic for macros
                return None;
            }
            let usage_node =
                name_ref.syntax().ancestors().find(|it| ast::PathExpr::can_cast(it.kind()));
            let usage_parent_option =
                usage_node.and_then(|it| it.parent()).and_then(ast::Expr::cast);
            let usage_parent = match usage_parent_option {
                Some(u) => u,
                None => return Some((range, name_ref, false)),
            };
            let initializer = matches!(
                initializer_expr,
                ast::Expr::CallExpr(_)
                    | ast::Expr::IndexExpr(_)
                    | ast::Expr::MethodCallExpr(_)
                    | ast::Expr::FieldExpr(_)
                    | ast::Expr::TryExpr(_)
                    | ast::Expr::Literal(_)
                    | ast::Expr::TupleExpr(_)
                    | ast::Expr::ArrayExpr(_)
                    | ast::Expr::ParenExpr(_)
                    | ast::Expr::PathExpr(_)
                    | ast::Expr::BlockExpr(_),
            );
            let parent = matches!(
                usage_parent,
                ast::Expr::CallExpr(_)
                    | ast::Expr::TupleExpr(_)
                    | ast::Expr::ArrayExpr(_)
                    | ast::Expr::ParenExpr(_)
                    | ast::Expr::ForExpr(_)
                    | ast::Expr::WhileExpr(_)
                    | ast::Expr::BreakExpr(_)
                    | ast::Expr::ReturnExpr(_)
                    | ast::Expr::MatchExpr(_)
                    | ast::Expr::BlockExpr(_)
            );
            Some((range, name_ref, !(initializer || parent)))
        })
        .collect::<Option<Vec<_>>>()?;

    let init_str = initializer_expr.syntax().text().to_string();
    let init_in_paren = format!("({init_str})");

    let target = match target {
        ast::NameOrNameRef::Name(it) => it.syntax().text_range(),
        ast::NameOrNameRef::NameRef(it) => it.syntax().text_range(),
    };

    acc.add(
        AssistId("inline_local_variable", AssistKind::RefactorInline),
        "Inline variable",
        target,
        move |builder| {
            if let Some(range) = delete_range {
                builder.delete(range);
            }
            for (range, name, should_wrap) in wrap_in_parens {
                let replacement = if should_wrap { &init_in_paren } else { &init_str };
                if ast::RecordExprField::for_field_name(&name).is_some() {
                    cov_mark::hit!(inline_field_shorthand);
                    builder.insert(range.end(), format!(": {replacement}"));
                } else {
                    builder.replace(range, replacement.clone())
                }
            }
        },
    )
}

struct InlineData {
    let_stmt: ast::LetStmt,
    delete_let: bool,
    target: ast::NameOrNameRef,
    references: Vec<FileReference>,
}

fn inline_let(
    sema: &Semantics<'_, RootDatabase>,
    let_stmt: ast::LetStmt,
    range: TextRange,
    file_id: FileId,
) -> Option<InlineData> {
    let bind_pat = match let_stmt.pat()? {
        ast::Pat::IdentPat(pat) => pat,
        _ => return None,
    };
    if bind_pat.mut_token().is_some() {
        cov_mark::hit!(test_not_inline_mut_variable);
        return None;
    }
    if !bind_pat.syntax().text_range().contains_range(range) {
        cov_mark::hit!(not_applicable_outside_of_bind_pat);
        return None;
    }

    let local = sema.to_def(&bind_pat)?;
    let UsageSearchResult { mut references } = Definition::Local(local).usages(sema).all();
    match references.remove(&file_id) {
        Some(references) => Some(InlineData {
            let_stmt,
            delete_let: true,
            target: ast::NameOrNameRef::Name(bind_pat.name()?),
            references,
        }),
        None => {
            cov_mark::hit!(test_not_applicable_if_variable_unused);
            None
        }
    }
}

fn inline_usage(
    sema: &Semantics<'_, RootDatabase>,
    path_expr: ast::PathExpr,
    range: TextRange,
    file_id: FileId,
) -> Option<InlineData> {
    let path = path_expr.path()?;
    let name = path.as_single_name_ref()?;
    if !name.syntax().text_range().contains_range(range) {
        cov_mark::hit!(test_not_inline_selection_too_broad);
        return None;
    }

    let local = match sema.resolve_path(&path)? {
        PathResolution::Local(local) => local,
        _ => return None,
    };
    if local.is_mut(sema.db) {
        cov_mark::hit!(test_not_inline_mut_variable_use);
        return None;
    }

    let sources = local.sources(sema.db);
    let [source] = sources.as_slice() else {
        // Not applicable with locals with multiple definitions (i.e. or patterns)
        return None;
    };

    let bind_pat = source.as_ident_pat()?;

    let let_stmt = ast::LetStmt::cast(bind_pat.syntax().parent()?)?;

    let UsageSearchResult { mut references } = Definition::Local(local).usages(sema).all();
    let mut references = references.remove(&file_id)?;
    let delete_let = references.len() == 1;
    references.retain(|fref| fref.name.as_name_ref() == Some(&name));

    Some(InlineData { let_stmt, delete_let, target: ast::NameOrNameRef::NameRef(name), references })
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_inline_let_bind_literal_expr() {
        check_assist(
            inline_local_variable,
            r"
fn bar(a: usize) {}
fn foo() {
    let a$0 = 1;
    a + 1;
    if a > 10 {
    }

    while a > 10 {

    }
    let b = a * 10;
    bar(a);
}",
            r"
fn bar(a: usize) {}
fn foo() {
    1 + 1;
    if 1 > 10 {
    }

    while 1 > 10 {

    }
    let b = 1 * 10;
    bar(1);
}",
        );
    }

    #[test]
    fn test_inline_let_bind_bin_expr() {
        check_assist(
            inline_local_variable,
            r"
fn bar(a: usize) {}
fn foo() {
    let a$0 = 1 + 1;
    a + 1;
    if a > 10 {
    }

    while a > 10 {

    }
    let b = a * 10;
    bar(a);
}",
            r"
fn bar(a: usize) {}
fn foo() {
    (1 + 1) + 1;
    if (1 + 1) > 10 {
    }

    while (1 + 1) > 10 {

    }
    let b = (1 + 1) * 10;
    bar(1 + 1);
}",
        );
    }

    #[test]
    fn test_inline_let_bind_function_call_expr() {
        check_assist(
            inline_local_variable,
            r"
fn bar(a: usize) {}
fn foo() {
    let a$0 = bar(1);
    a + 1;
    if a > 10 {
    }

    while a > 10 {

    }
    let b = a * 10;
    bar(a);
}",
            r"
fn bar(a: usize) {}
fn foo() {
    bar(1) + 1;
    if bar(1) > 10 {
    }

    while bar(1) > 10 {

    }
    let b = bar(1) * 10;
    bar(bar(1));
}",
        );
    }

    #[test]
    fn test_inline_let_bind_cast_expr() {
        check_assist(
            inline_local_variable,
            r"
fn bar(a: usize): usize { a }
fn foo() {
    let a$0 = bar(1) as u64;
    a + 1;
    if a > 10 {
    }

    while a > 10 {

    }
    let b = a * 10;
    bar(a);
}",
            r"
fn bar(a: usize): usize { a }
fn foo() {
    (bar(1) as u64) + 1;
    if (bar(1) as u64) > 10 {
    }

    while (bar(1) as u64) > 10 {

    }
    let b = (bar(1) as u64) * 10;
    bar(bar(1) as u64);
}",
        );
    }

    #[test]
    fn test_inline_let_bind_block_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let a$0 = { 10 + 1 };
    a + 1;
    if a > 10 {
    }

    while a > 10 {

    }
    let b = a * 10;
    bar(a);
}",
            r"
fn foo() {
    { 10 + 1 } + 1;
    if { 10 + 1 } > 10 {
    }

    while { 10 + 1 } > 10 {

    }
    let b = { 10 + 1 } * 10;
    bar({ 10 + 1 });
}",
        );
    }

    #[test]
    fn test_inline_let_bind_paren_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let a$0 = ( 10 + 1 );
    a + 1;
    if a > 10 {
    }

    while a > 10 {

    }
    let b = a * 10;
    bar(a);
}",
            r"
fn foo() {
    ( 10 + 1 ) + 1;
    if ( 10 + 1 ) > 10 {
    }

    while ( 10 + 1 ) > 10 {

    }
    let b = ( 10 + 1 ) * 10;
    bar(( 10 + 1 ));
}",
        );
    }

    #[test]
    fn test_not_inline_mut_variable() {
        cov_mark::check!(test_not_inline_mut_variable);
        check_assist_not_applicable(
            inline_local_variable,
            r"
fn foo() {
    let mut a$0 = 1 + 1;
    a + 1;
}",
        );
    }

    #[test]
    fn test_not_inline_mut_variable_use() {
        cov_mark::check!(test_not_inline_mut_variable_use);
        check_assist_not_applicable(
            inline_local_variable,
            r"
fn foo() {
    let mut a = 1 + 1;
    a$0 + 1;
}",
        );
    }

    #[test]
    fn test_call_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let a$0 = bar(10 + 1);
    let b = a * 10;
    let c = a as usize;
}",
            r"
fn foo() {
    let b = bar(10 + 1) * 10;
    let c = bar(10 + 1) as usize;
}",
        );
    }

    #[test]
    fn test_index_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let x = vec![1, 2, 3];
    let a$0 = x[0];
    let b = a * 10;
    let c = a as usize;
}",
            r"
fn foo() {
    let x = vec![1, 2, 3];
    let b = x[0] * 10;
    let c = x[0] as usize;
}",
        );
    }

    #[test]
    fn test_method_call_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let bar = vec![1];
    let a$0 = bar.len();
    let b = a * 10;
    let c = a as usize;
}",
            r"
fn foo() {
    let bar = vec![1];
    let b = bar.len() * 10;
    let c = bar.len() as usize;
}",
        );
    }

    #[test]
    fn test_field_expr() {
        check_assist(
            inline_local_variable,
            r"
struct Bar {
    foo: usize
}

fn foo() {
    let bar = Bar { foo: 1 };
    let a$0 = bar.foo;
    let b = a * 10;
    let c = a as usize;
}",
            r"
struct Bar {
    foo: usize
}

fn foo() {
    let bar = Bar { foo: 1 };
    let b = bar.foo * 10;
    let c = bar.foo as usize;
}",
        );
    }

    #[test]
    fn test_try_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() -> Option<usize> {
    let bar = Some(1);
    let a$0 = bar?;
    let b = a * 10;
    let c = a as usize;
    None
}",
            r"
fn foo() -> Option<usize> {
    let bar = Some(1);
    let b = bar? * 10;
    let c = bar? as usize;
    None
}",
        );
    }

    #[test]
    fn test_ref_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let bar = 10;
    let a$0 = &bar;
    let b = a * 10;
}",
            r"
fn foo() {
    let bar = 10;
    let b = (&bar) * 10;
}",
        );
    }

    #[test]
    fn test_tuple_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let a$0 = (10, 20);
    let b = a[0];
}",
            r"
fn foo() {
    let b = (10, 20)[0];
}",
        );
    }

    #[test]
    fn test_array_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let a$0 = [1, 2, 3];
    let b = a.len();
}",
            r"
fn foo() {
    let b = [1, 2, 3].len();
}",
        );
    }

    #[test]
    fn test_paren() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let a$0 = (10 + 20);
    let b = a * 10;
    let c = a as usize;
}",
            r"
fn foo() {
    let b = (10 + 20) * 10;
    let c = (10 + 20) as usize;
}",
        );
    }

    #[test]
    fn test_path_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let d = 10;
    let a$0 = d;
    let b = a * 10;
    let c = a as usize;
}",
            r"
fn foo() {
    let d = 10;
    let b = d * 10;
    let c = d as usize;
}",
        );
    }

    #[test]
    fn test_block_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let a$0 = { 10 };
    let b = a * 10;
    let c = a as usize;
}",
            r"
fn foo() {
    let b = { 10 } * 10;
    let c = { 10 } as usize;
}",
        );
    }

    #[test]
    fn test_used_in_different_expr1() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let a$0 = 10 + 20;
    let b = a * 10;
    let c = (a, 20);
    let d = [a, 10];
    let e = (a);
}",
            r"
fn foo() {
    let b = (10 + 20) * 10;
    let c = (10 + 20, 20);
    let d = [10 + 20, 10];
    let e = (10 + 20);
}",
        );
    }

    #[test]
    fn test_used_in_for_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let a$0 = vec![10, 20];
    for i in a {}
}",
            r"
fn foo() {
    for i in vec![10, 20] {}
}",
        );
    }

    #[test]
    fn test_used_in_while_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let a$0 = 1 > 0;
    while a {}
}",
            r"
fn foo() {
    while 1 > 0 {}
}",
        );
    }

    #[test]
    fn test_used_in_break_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let a$0 = 1 + 1;
    loop {
        break a;
    }
}",
            r"
fn foo() {
    loop {
        break 1 + 1;
    }
}",
        );
    }

    #[test]
    fn test_used_in_return_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let a$0 = 1 > 0;
    return a;
}",
            r"
fn foo() {
    return 1 > 0;
}",
        );
    }

    #[test]
    fn test_used_in_match_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let a$0 = 1 > 0;
    match a {}
}",
            r"
fn foo() {
    match 1 > 0 {}
}",
        );
    }

    #[test]
    fn inline_field_shorthand() {
        cov_mark::check!(inline_field_shorthand);
        check_assist(
            inline_local_variable,
            r"
struct S { foo: i32}
fn main() {
    let $0foo = 92;
    S { foo }
}
",
            r"
struct S { foo: i32}
fn main() {
    S { foo: 92 }
}
",
        );
    }

    #[test]
    fn test_not_applicable_if_variable_unused() {
        cov_mark::check!(test_not_applicable_if_variable_unused);
        check_assist_not_applicable(
            inline_local_variable,
            r"
fn foo() {
    let $0a = 0;
}
            ",
        )
    }

    #[test]
    fn not_applicable_outside_of_bind_pat() {
        cov_mark::check!(not_applicable_outside_of_bind_pat);
        check_assist_not_applicable(
            inline_local_variable,
            r"
fn main() {
    let x = $01 + 2;
    x * 4;
}
",
        )
    }

    #[test]
    fn works_on_local_usage() {
        check_assist(
            inline_local_variable,
            r#"
fn f() {
    let xyz = 0;
    xyz$0;
}
"#,
            r#"
fn f() {
    0;
}
"#,
        );
    }

    #[test]
    fn does_not_remove_let_when_multiple_usages() {
        check_assist(
            inline_local_variable,
            r#"
fn f() {
    let xyz = 0;
    xyz$0;
    xyz;
}
"#,
            r#"
fn f() {
    let xyz = 0;
    0;
    xyz;
}
"#,
        );
    }

    #[test]
    fn not_applicable_with_non_ident_pattern() {
        check_assist_not_applicable(
            inline_local_variable,
            r#"
fn main() {
    let (x, y) = (0, 1);
    x$0;
}
"#,
        );
    }

    #[test]
    fn not_applicable_on_local_usage_in_macro() {
        check_assist_not_applicable(
            inline_local_variable,
            r#"
macro_rules! m {
    ($i:ident) => { $i }
}
fn f() {
    let xyz = 0;
    m!(xyz$0); // replacing it would break the macro
}
"#,
        );
        check_assist_not_applicable(
            inline_local_variable,
            r#"
macro_rules! m {
    ($i:ident) => { $i }
}
fn f() {
    let xyz$0 = 0;
    m!(xyz); // replacing it would break the macro
}
"#,
        );
    }

    #[test]
    fn test_not_inline_selection_too_broad() {
        cov_mark::check!(test_not_inline_selection_too_broad);
        check_assist_not_applicable(
            inline_local_variable,
            r#"
fn f() {
    let foo = 0;
    let bar = 0;
    $0foo + bar$0;
}
"#,
        );
    }

    #[test]
    fn test_inline_ref_in_let() {
        check_assist(
            inline_local_variable,
            r#"
fn f() {
    let x = {
        let y = 0;
        y$0
    };
}
"#,
            r#"
fn f() {
    let x = {
        0
    };
}
"#,
        );
    }

    #[test]
    fn test_inline_let_unit_struct() {
        check_assist_not_applicable(
            inline_local_variable,
            r#"
struct S;
fn f() {
    let S$0 = S;
    S;
}
"#,
        );
    }
}
