use itertools::Itertools;
use rustc_hash::FxHashMap;

use hir::{Adt, ModuleDef, PathResolution, Semantics, Struct};
use ide_db::RootDatabase;
use syntax::{algo, ast, match_ast, AstNode, SyntaxKind, SyntaxKind::*, SyntaxNode};
use test_utils::mark;

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: reorder_fields
//
// Reorder the fields of record literals and record patterns in the same order as in
// the definition.
//
// ```
// struct Foo {foo: i32, bar: i32};
// const test: Foo = $0Foo {bar: 0, foo: 1}
// ```
// ->
// ```
// struct Foo {foo: i32, bar: i32};
// const test: Foo = Foo {foo: 1, bar: 0}
// ```
//
pub(crate) fn reorder_fields(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    reorder::<ast::RecordExpr>(acc, ctx).or_else(|| reorder::<ast::RecordPat>(acc, ctx))
}

fn reorder<R: AstNode>(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let record = ctx.find_node_at_offset::<R>()?;
    let path = record.syntax().children().find_map(ast::Path::cast)?;

    let ranks = compute_fields_ranks(&path, &ctx)?;

    let fields = get_fields(&record.syntax());
    let sorted_fields = sorted_by_rank(&fields, |node| {
        *ranks.get(&get_field_name(node)).unwrap_or(&usize::max_value())
    });

    if sorted_fields == fields {
        mark::hit!(reorder_sorted_fields);
        return None;
    }

    let target = record.syntax().text_range();
    acc.add(
        AssistId("reorder_fields", AssistKind::RefactorRewrite),
        "Reorder record fields",
        target,
        |edit| {
            let mut rewriter = algo::SyntaxRewriter::default();
            for (old, new) in fields.iter().zip(&sorted_fields) {
                rewriter.replace(old, new);
            }
            edit.rewrite(rewriter);
        },
    )
}

fn get_fields_kind(node: &SyntaxNode) -> Vec<SyntaxKind> {
    match node.kind() {
        RECORD_EXPR => vec![RECORD_EXPR_FIELD],
        RECORD_PAT => vec![RECORD_PAT_FIELD, IDENT_PAT],
        _ => vec![],
    }
}

fn get_field_name(node: &SyntaxNode) -> String {
    let res = match_ast! {
        match node {
            ast::RecordExprField(field) => field.field_name().map(|it| it.to_string()),
            ast::RecordPatField(field) => field.field_name().map(|it| it.to_string()),
            _ => None,
        }
    };
    res.unwrap_or_default()
}

fn get_fields(record: &SyntaxNode) -> Vec<SyntaxNode> {
    let kinds = get_fields_kind(record);
    record.children().flat_map(|n| n.children()).filter(|n| kinds.contains(&n.kind())).collect()
}

fn sorted_by_rank(
    fields: &[SyntaxNode],
    get_rank: impl Fn(&SyntaxNode) -> usize,
) -> Vec<SyntaxNode> {
    fields.iter().cloned().sorted_by_key(get_rank).collect()
}

fn struct_definition(path: &ast::Path, sema: &Semantics<RootDatabase>) -> Option<Struct> {
    match sema.resolve_path(path) {
        Some(PathResolution::Def(ModuleDef::Adt(Adt::Struct(s)))) => Some(s),
        _ => None,
    }
}

fn compute_fields_ranks(path: &ast::Path, ctx: &AssistContext) -> Option<FxHashMap<String, usize>> {
    Some(
        struct_definition(path, &ctx.sema)?
            .fields(ctx.db())
            .iter()
            .enumerate()
            .map(|(idx, field)| (field.name(ctx.db()).to_string(), idx))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use test_utils::mark;

    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn reorder_sorted_fields() {
        mark::check!(reorder_sorted_fields);
        check_assist_not_applicable(
            reorder_fields,
            r#"
struct Foo {
    foo: i32,
    bar: i32,
}

const test: Foo = $0Foo { foo: 0, bar: 0 };
"#,
        )
    }

    #[test]
    fn trivial_empty_fields() {
        check_assist_not_applicable(
            reorder_fields,
            r#"
struct Foo {};
const test: Foo = $0Foo {}
"#,
        )
    }

    #[test]
    fn reorder_struct_fields() {
        check_assist(
            reorder_fields,
            r#"
struct Foo {foo: i32, bar: i32};
const test: Foo = $0Foo {bar: 0, foo: 1}
"#,
            r#"
struct Foo {foo: i32, bar: i32};
const test: Foo = Foo {foo: 1, bar: 0}
"#,
        )
    }

    #[test]
    fn reorder_struct_pattern() {
        check_assist(
            reorder_fields,
            r#"
struct Foo { foo: i64, bar: i64, baz: i64 }

fn f(f: Foo) -> {
    match f {
        $0Foo { baz: 0, ref mut bar, .. } => (),
        _ => ()
    }
}
"#,
            r#"
struct Foo { foo: i64, bar: i64, baz: i64 }

fn f(f: Foo) -> {
    match f {
        Foo { ref mut bar, baz: 0, .. } => (),
        _ => ()
    }
}
"#,
        )
    }

    #[test]
    fn reorder_with_extra_field() {
        check_assist(
            reorder_fields,
            r#"
struct Foo {
    foo: String,
    bar: String,
}

impl Foo {
    fn new() -> Foo {
        let foo = String::new();
        $0Foo {
            bar: foo.clone(),
            extra: "Extra field",
            foo,
        }
    }
}
"#,
            r#"
struct Foo {
    foo: String,
    bar: String,
}

impl Foo {
    fn new() -> Foo {
        let foo = String::new();
        Foo {
            foo,
            bar: foo.clone(),
            extra: "Extra field",
        }
    }
}
"#,
        )
    }
}
