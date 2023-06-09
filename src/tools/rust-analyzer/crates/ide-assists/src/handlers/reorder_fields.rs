use either::Either;
use ide_db::FxHashMap;
use itertools::Itertools;
use syntax::{ast, ted, AstNode};

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
pub(crate) fn reorder_fields(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let path = ctx.find_node_at_offset::<ast::Path>()?;
    let record =
        path.syntax().parent().and_then(<Either<ast::RecordExpr, ast::RecordPat>>::cast)?;

    let ranks = compute_fields_ranks(&path, ctx)?;
    let get_rank_of_field =
        |of: Option<_>| *ranks.get(&of.unwrap_or_default()).unwrap_or(&usize::MAX);

    let field_list = match &record {
        Either::Left(it) => Either::Left(it.record_expr_field_list()?),
        Either::Right(it) => Either::Right(it.record_pat_field_list()?),
    };
    let fields = match field_list {
        Either::Left(it) => Either::Left((
            it.fields()
                .sorted_unstable_by_key(|field| {
                    get_rank_of_field(field.field_name().map(|it| it.to_string()))
                })
                .collect::<Vec<_>>(),
            it,
        )),
        Either::Right(it) => Either::Right((
            it.fields()
                .sorted_unstable_by_key(|field| {
                    get_rank_of_field(field.field_name().map(|it| it.to_string()))
                })
                .collect::<Vec<_>>(),
            it,
        )),
    };

    let is_sorted = fields.as_ref().either(
        |(sorted, field_list)| field_list.fields().zip(sorted).all(|(a, b)| a == *b),
        |(sorted, field_list)| field_list.fields().zip(sorted).all(|(a, b)| a == *b),
    );
    if is_sorted {
        cov_mark::hit!(reorder_sorted_fields);
        return None;
    }
    let target = record.as_ref().either(AstNode::syntax, AstNode::syntax).text_range();
    acc.add(
        AssistId("reorder_fields", AssistKind::RefactorRewrite),
        "Reorder record fields",
        target,
        |builder| match fields {
            Either::Left((sorted, field_list)) => {
                replace(builder.make_mut(field_list).fields(), sorted)
            }
            Either::Right((sorted, field_list)) => {
                replace(builder.make_mut(field_list).fields(), sorted)
            }
        },
    )
}

fn replace<T: AstNode + PartialEq>(
    fields: impl Iterator<Item = T>,
    sorted_fields: impl IntoIterator<Item = T>,
) {
    fields.zip(sorted_fields).for_each(|(field, sorted_field)| {
        ted::replace(field.syntax(), sorted_field.syntax().clone_for_update())
    });
}

fn compute_fields_ranks(
    path: &ast::Path,
    ctx: &AssistContext<'_>,
) -> Option<FxHashMap<String, usize>> {
    let strukt = match ctx.sema.resolve_path(path) {
        Some(hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Struct(it)))) => it,
        _ => return None,
    };

    let res = strukt
        .fields(ctx.db())
        .into_iter()
        .enumerate()
        .map(|(idx, field)| (field.name(ctx.db()).display(ctx.db()).to_string(), idx))
        .collect();

    Some(res)
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn reorder_sorted_fields() {
        cov_mark::check!(reorder_sorted_fields);
        check_assist_not_applicable(
            reorder_fields,
            r#"
struct Foo { foo: i32, bar: i32 }
const test: Foo = $0Foo { foo: 0, bar: 0 };
"#,
        )
    }

    #[test]
    fn trivial_empty_fields() {
        check_assist_not_applicable(
            reorder_fields,
            r#"
struct Foo {}
const test: Foo = $0Foo {};
"#,
        )
    }

    #[test]
    fn reorder_struct_fields() {
        check_assist(
            reorder_fields,
            r#"
struct Foo { foo: i32, bar: i32 }
const test: Foo = $0Foo { bar: 0, foo: 1 };
"#,
            r#"
struct Foo { foo: i32, bar: i32 }
const test: Foo = Foo { foo: 1, bar: 0 };
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
struct Foo { foo: String, bar: String }

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
struct Foo { foo: String, bar: String }

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
