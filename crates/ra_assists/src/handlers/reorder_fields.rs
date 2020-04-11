use std::collections::HashMap;

use itertools::Itertools;

use hir::{Adt, ModuleDef, PathResolution, Semantics, Struct};
use ra_ide_db::RootDatabase;
use ra_syntax::ast::{Name, Pat};
use ra_syntax::{
    ast,
    ast::{Path, RecordField, RecordLit, RecordPat},
    AstNode,
};

use crate::{
    assist_ctx::{Assist, AssistCtx},
    AssistId,
};

pub(crate) fn reorder_fields(ctx: AssistCtx) -> Option<Assist> {
    reorder_struct(ctx.clone()).or_else(|| reorder_struct_pat(ctx))
}

fn reorder_struct(ctx: AssistCtx) -> Option<Assist> {
    let record: RecordLit = ctx.find_node_at_offset()?;
    reorder(ctx, &record, &record.path()?, field_name)
}

fn field_name(r: &RecordField) -> String {
    r.name_ref()
        .map(|name| name.syntax().text().to_string())
        .or_else(|| r.expr().map(|e| e.syntax().text().to_string()))
        .unwrap_or_default()
}

fn reorder_struct_pat(ctx: AssistCtx) -> Option<Assist> {
    let record: RecordPat = ctx.find_node_at_offset()?;
    reorder(ctx, &record, &record.path()?, field_pat_name)
}

fn field_pat_name(field: &Pat) -> String {
    field.syntax().children().find_map(Name::cast).map(|n| n.to_string()).unwrap_or_default()
}

fn reorder<R: AstNode, F: AstNode + Eq + Clone>(
    ctx: AssistCtx,
    record: &R,
    path: &Path,
    field_name: fn(&F) -> String,
) -> Option<Assist> {
    let ranks = compute_fields_ranks(path, &ctx)?;
    let fields: Vec<F> = get_fields(record);
    let sorted_fields: Vec<F> =
        sort_by_rank(&fields, |f| *ranks.get(&field_name(f)).unwrap_or(&usize::max_value()));

    if sorted_fields == fields {
        return None;
    }

    ctx.add_assist(AssistId("reorder_fields"), "Reorder record fields", |edit| {
        for (old, new) in fields.into_iter().zip(sorted_fields) {
            edit.replace_ast(old, new);
        }
        edit.target(record.syntax().text_range())
    })
}

fn get_fields<R: AstNode, F: AstNode>(record: &R) -> Vec<F> {
    record.syntax().children().flat_map(|n1| n1.children()).filter_map(|n3| F::cast(n3)).collect()
}

fn sort_by_rank<F: AstNode + Clone>(fields: &[F], get_rank: impl FnMut(&F) -> usize) -> Vec<F> {
    fields.iter().cloned().sorted_by_key(get_rank).collect()
}

fn struct_definition(path: &ast::Path, sema: &Semantics<RootDatabase>) -> Option<Struct> {
    match sema.resolve_path(path) {
        Some(PathResolution::Def(ModuleDef::Adt(Adt::Struct(s)))) => Some(s),
        _ => None,
    }
}

fn compute_fields_ranks(path: &Path, ctx: &AssistCtx) -> Option<HashMap<String, usize>> {
    Some(
        struct_definition(path, ctx.sema)?
            .fields(ctx.db)
            .iter()
            .enumerate()
            .map(|(idx, field)| (field.name(ctx.db).to_string(), idx))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use crate::helpers::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn not_applicable_if_sorted() {
        check_assist_not_applicable(
            reorder_fields,
            r#"
        struct Foo {
            foo: i32,
            bar: i32,
        }

        const test: Foo = <|>Foo { foo: 0, bar: 0 };
        "#,
        )
    }

    #[test]
    fn trivial_empty_fields() {
        check_assist_not_applicable(
            reorder_fields,
            r#"
        struct Foo {};
        const test: Foo = <|>Foo {}
        "#,
        )
    }

    #[test]
    fn reorder_struct_fields() {
        check_assist(
            reorder_fields,
            r#"
        struct Foo {foo: i32, bar: i32};
        const test: Foo = <|>Foo {bar: 0, foo: 1}
        "#,
            r#"
        struct Foo {foo: i32, bar: i32};
        const test: Foo = <|>Foo {foo: 1, bar: 0}
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
                <|>Foo { baz: 0, ref mut bar, .. } => (),
                _ => ()
            }
        }
        "#,
            r#"
        struct Foo { foo: i64, bar: i64, baz: i64 }

        fn f(f: Foo) -> {
            match f {
                <|>Foo { ref mut bar, baz: 0, .. } => (),
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
                    <|>Foo {
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
                    <|>Foo {
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
