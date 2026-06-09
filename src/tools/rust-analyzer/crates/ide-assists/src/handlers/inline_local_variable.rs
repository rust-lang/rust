use either::{Either, for_both};
use hir::{PathResolution, Semantics};
use ide_db::{
    RootDatabase,
    defs::Definition,
    search::{FileReference, UsageSearchResult},
};
use syntax::{
    Direction, T, TextRange,
    ast::{self, AstNode, AstToken, HasName},
    syntax_editor::{Element, Position, SyntaxEditor},
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists},
    utils::cover_edit_range,
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
pub(crate) fn inline_local_variable(acc: &mut Assists, ctx: &AssistContext<'_, '_>) -> Option<()> {
    let source = ctx.source_file().syntax();
    let range = ctx.selection_trimmed();
    let InlineData { let_stmt, delete_let, references, target } =
        if let Some(path_expr) = ctx.find_node_at_offset_with_descend::<ast::PathExpr>() {
            inline_usage(&ctx.sema, path_expr, range)
        } else if let Some(let_stmt) = ctx.find_node_at_offset() {
            inline_let(&ctx.sema, let_stmt, range)
        } else {
            None
        }?;
    let initializer_expr = match &let_stmt {
        either::Either::Left(it) => it.initializer()?,
        either::Either::Right(it) => it.expr()?,
    };
    let needs_parens = |name_ref: &ast::NameRef| {
        let usage_node =
            name_ref.syntax().ancestors().find(|it| ast::PathExpr::can_cast(it.kind()));
        let usage_parent = usage_node.as_ref().and_then(|it| it.parent());
        match (usage_node, usage_parent) {
            (Some(usage), Some(parent)) => {
                initializer_expr.needs_parens_in_place_of(&parent, &usage)
            }
            _ => false,
        }
    };

    acc.add(
        AssistId::refactor_inline("inline_local_variable"),
        "Inline variable",
        target.range,
        |builder| {
            let editor = builder.make_editor(source);
            let make = editor.make();
            if delete_let {
                editor.delete(let_stmt.syntax());

                // Processing let-expr in let-chain
                if let Some(bin_expr) = let_stmt.syntax().parent().and_then(ast::BinExpr::cast)
                    && let Some(op_token) = bin_expr.op_token()
                {
                    editor.delete(&op_token);
                    remove_whitespace(op_token, Direction::Prev, &editor);
                    remove_whitespace(let_stmt.syntax(), Direction::Prev, &editor);
                } else {
                    remove_whitespace(let_stmt.syntax(), Direction::Next, &editor);
                }
            }

            for FileReference { range, name, .. } in references {
                let Some(name) = name.as_name_ref().cloned() else { continue };
                let replacement = if needs_parens(&name) {
                    make.expr_paren(initializer_expr.clone()).into()
                } else {
                    initializer_expr.clone()
                };

                let place = cover_edit_range(source, range);
                if ast::RecordExprField::for_field_name(&name).is_some() {
                    cov_mark::hit!(inline_field_shorthand);
                    editor.insert_all(
                        Position::after(place.end()),
                        vec![
                            make.token(T![:]).into(),
                            make.whitespace(" ").into(),
                            replacement.syntax().clone().into(),
                        ],
                    );
                } else {
                    editor.replace_all(place, vec![replacement.syntax().clone().into()]);
                }
            }
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

struct InlineData {
    let_stmt: Either<ast::LetStmt, ast::LetExpr>,
    delete_let: bool,
    target: hir::FileRange,
    references: Vec<FileReference>,
}

fn inline_let(
    sema: &Semantics<'_, RootDatabase>,
    let_stmt: Either<ast::LetStmt, ast::LetExpr>,
    range: TextRange,
) -> Option<InlineData> {
    let bind_pat = match for_both!(&let_stmt, it => it.pat())? {
        ast::Pat::IdentPat(pat) => pat,
        _ => return None,
    };
    if bind_pat.mut_token().is_some() {
        cov_mark::hit!(test_not_inline_mut_variable);
        return None;
    }
    let target = sema.original_range_opt(bind_pat.name()?.syntax())?;
    if !target.range.contains_range(range) {
        cov_mark::hit!(not_applicable_outside_of_bind_pat);
        return None;
    }

    let local = sema.to_def(&bind_pat)?;
    let UsageSearchResult { references } = Definition::Local(local).usages(sema).all();
    let references = references.into_iter().flat_map(|it| it.1).collect::<Vec<_>>();

    match references.first() {
        Some(_) => Some(InlineData { let_stmt, delete_let: true, target, references }),
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
) -> Option<InlineData> {
    let path = path_expr.path()?;
    let name = path.as_single_name_ref()?;
    let target = sema.original_range_opt(name.syntax())?;
    if !target.range.contains_range(range) {
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

    let let_stmt = AstNode::cast(bind_pat.syntax().parent()?)?;

    let UsageSearchResult { references } = Definition::Local(local).usages(sema).all();
    let mut references = references.into_iter().flat_map(|it| it.1).collect::<Vec<_>>();
    let delete_let = references.len() == 1;
    references.retain(|fref| fref.name.as_name_ref() == Some(&name));

    Some(InlineData { let_stmt, delete_let, target, references })
}

fn remove_whitespace(elem: impl Element, dir: Direction, editor: &SyntaxEditor) {
    let token = match elem.syntax_element() {
        syntax::NodeOrToken::Node(node) => match dir {
            Direction::Next => node.last_token(),
            Direction::Prev => node.first_token(),
        },
        syntax::NodeOrToken::Token(t) => Some(t),
    };
    let next_token = match dir {
        Direction::Next => token.and_then(|it| it.next_token()),
        Direction::Prev => token.and_then(|it| it.prev_token()),
    };
    if let Some(whitespace) = next_token.and_then(ast::Whitespace::cast) {
        editor.delete(whitespace.syntax());
    }
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
    1 + 1 + 1;
    if 1 + 1 > 10 {
    }

    while 1 + 1 > 10 {

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
//- minicore: sized
fn bar(a: usize) -> usize { a }
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
fn bar(a: usize) -> usize { a }
fn foo() {
    bar(1) as u64 + 1;
    if bar(1) as u64 > 10 {
    }

    while bar(1) as u64 > 10 {

    }
    let b = bar(1) as u64 * 10;
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
    fn test_inline_let_expr() {
        check_assist(
            inline_local_variable,
            r"
fn bar(a: usize) {}
fn foo() {
    if let a$0 = 1
        && true
    {
        a + 1;
        if a > 10 {}
        while a > 10 {}
        let b = a * 10;
        bar(a);
    }
}",
            r"
fn bar(a: usize) {}
fn foo() {
    if true
    {
        1 + 1;
        if 1 > 10 {}
        while 1 > 10 {}
        let b = 1 * 10;
        bar(1);
    }
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
    let b = &bar * 10;
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
//- minicore: iterator
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
    fn let_expr_works_on_local_usage() {
        check_assist(
            inline_local_variable,
            r#"
fn f() {
    if let xyz = 0
        && true
    {
        xyz$0;
    }
}
"#,
            r#"
fn f() {
    if true
    {
        0;
    }
}
"#,
        );

        check_assist(
            inline_local_variable,
            r#"
fn f() {
    if let xyz = true
        && xyz$0
    {
    }
}
"#,
            r#"
fn f() {
    if true
    {
    }
}
"#,
        );

        check_assist(
            inline_local_variable,
            r#"
fn f() {
    if true
        && let xyz = 0
    {
        xyz$0;
    }
}
"#,
            r#"
fn f() {
    if true
    {
        0;
    }
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
    fn local_usage_in_macro() {
        check_assist(
            inline_local_variable,
            r#"
macro_rules! m {
    ($i:ident) => { $i }
}
fn f() {
    let xyz = 0;
    m!(xyz$0); // some macros may break, but it's best to support them
}
"#,
            r#"
macro_rules! m {
    ($i:ident) => { $i }
}
fn f() {
    m!(0); // some macros may break, but it's best to support them
}
"#,
        );
        check_assist(
            inline_local_variable,
            r#"
macro_rules! m {
    ($i:ident) => { $i }
}
fn f() {
    let xyz$0 = 0;
    m!(xyz); // some macros may break, but it's best to support them
}
"#,
            r#"
macro_rules! m {
    ($i:ident) => { $i }
}
fn f() {
    m!(0); // some macros may break, but it's best to support them
}
"#,
        );
        // FIXME: supports let-stmt inside macro case
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

    #[test]
    fn test_inline_closure() {
        check_assist(
            inline_local_variable,
            r#"
//- minicore: fn
fn main() {
    let $0f = || 2;
    let _ = f();
}
"#,
            r#"
fn main() {
    let _ = (|| 2)();
}
"#,
        );
    }

    #[test]
    fn test_wrap_in_parens() {
        check_assist(
            inline_local_variable,
            r#"
fn main() {
    let $0a = 123 < 456;
    let b = !a;
}
"#,
            r#"
fn main() {
    let b = !(123 < 456);
}
"#,
        );
        check_assist(
            inline_local_variable,
            r#"
trait Foo {
    fn foo(&self);
}

impl Foo for bool {
    fn foo(&self) {}
}

fn main() {
    let $0a = 123 < 456;
    let b = a.foo();
}
"#,
            r#"
trait Foo {
    fn foo(&self);
}

impl Foo for bool {
    fn foo(&self) {}
}

fn main() {
    let b = (123 < 456).foo();
}
"#,
        );
    }
}
