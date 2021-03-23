use rustc_hash::FxHashMap;

use syntax::{algo, ast, match_ast, AstNode, SyntaxKind::*, SyntaxNode};

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
    let record = ctx
        .find_node_at_offset::<ast::RecordExpr>()
        .map(|it| it.syntax().clone())
        .or_else(|| ctx.find_node_at_offset::<ast::RecordPat>().map(|it| it.syntax().clone()))?;

    let path = record.children().find_map(ast::Path::cast)?;

    let ranks = compute_fields_ranks(&path, &ctx)?;

    let fields: Vec<SyntaxNode> = {
        let field_kind = match record.kind() {
            RECORD_EXPR => RECORD_EXPR_FIELD,
            RECORD_PAT => RECORD_PAT_FIELD,
            _ => {
                stdx::never!();
                return None;
            }
        };
        record.children().flat_map(|n| n.children()).filter(|n| n.kind() == field_kind).collect()
    };

    let sorted_fields = {
        let mut fields = fields.clone();
        fields.sort_by_key(|node| *ranks.get(&get_field_name(node)).unwrap_or(&usize::max_value()));
        fields
    };

    if sorted_fields == fields {
        cov_mark::hit!(reorder_sorted_fields);
        return None;
    }

    let target = record.text_range();
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

fn compute_fields_ranks(path: &ast::Path, ctx: &AssistContext) -> Option<FxHashMap<String, usize>> {
    let strukt = match ctx.sema.resolve_path(path) {
        Some(hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Struct(it)))) => it,
        _ => return None,
    };

    let res = strukt
        .fields(ctx.db())
        .iter()
        .enumerate()
        .map(|(idx, field)| (field.name(ctx.db()).to_string(), idx))
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
