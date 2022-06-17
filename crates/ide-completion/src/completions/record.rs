//! Complete fields in record literals and patterns.
use ide_db::SymbolKind;
use syntax::{ast::Expr, T};

use crate::{
    context::{NameRefContext, NameRefKind, PathCompletionCtx, PathKind, PatternContext},
    CompletionContext, CompletionItem, CompletionItemKind, CompletionRelevance,
    CompletionRelevancePostfixMatch, Completions,
};

pub(crate) fn complete_record(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    let missing_fields = if let Some(PatternContext { record_pat: Some(record_pat), .. }) =
        &ctx.pattern_ctx
    {
        ctx.sema.record_pattern_missing_fields(record_pat)
    } else if let Some(NameRefContext {
        kind:
            Some(
                NameRefKind::RecordExpr(record_expr)
                | NameRefKind::Path(PathCompletionCtx {
                    kind: PathKind::Expr { is_func_update: Some(record_expr), .. },
                    qualifier: None,
                    ..
                }),
            ),
        ..
    }) = ctx.nameref_ctx()
    {
        let ty = ctx.sema.type_of_expr(&Expr::RecordExpr(record_expr.clone()));

        if let Some(hir::Adt::Union(un)) = ty.as_ref().and_then(|t| t.original.as_adt()) {
            // ctx.sema.record_literal_missing_fields will always return
            // an empty Vec on a union literal. This is normally
            // reasonable, but here we'd like to present the full list
            // of fields if the literal is empty.
            let were_fields_specified =
                record_expr.record_expr_field_list().and_then(|fl| fl.fields().next()).is_some();

            match were_fields_specified {
                false => un.fields(ctx.db).into_iter().map(|f| (f, f.ty(ctx.db))).collect(),
                true => vec![],
            }
        } else {
            let missing_fields = ctx.sema.record_literal_missing_fields(record_expr);

            let default_trait = ctx.famous_defs().core_default_Default();
            let impl_default_trait =
                default_trait.zip(ty.as_ref()).map_or(false, |(default_trait, ty)| {
                    ty.original.impls_trait(ctx.db, default_trait, &[])
                });

            if impl_default_trait && !missing_fields.is_empty() {
                let completion_text = "..Default::default()";
                let mut item =
                    CompletionItem::new(SymbolKind::Field, ctx.source_range(), completion_text);
                let completion_text =
                    completion_text.strip_prefix(ctx.token.text()).unwrap_or(completion_text);
                item.insert_text(completion_text).set_relevance(CompletionRelevance {
                    postfix_match: Some(CompletionRelevancePostfixMatch::Exact),
                    ..Default::default()
                });
                item.add_to(acc);
            }
            if ctx.previous_token_is(T![.]) {
                let mut item =
                    CompletionItem::new(CompletionItemKind::Snippet, ctx.source_range(), "..");
                item.insert_text(".");
                item.add_to(acc);
                return None;
            }
            missing_fields
        }
    } else {
        return None;
    };

    for (field, ty) in missing_fields {
        acc.add_field(ctx, None, field, &ty);
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use crate::tests::check_edit;

    #[test]
    fn literal_struct_completion_edit() {
        check_edit(
            "FooDesc {…}",
            r#"
struct FooDesc { pub bar: bool }

fn create_foo(foo_desc: &FooDesc) -> () { () }

fn baz() {
    let foo = create_foo(&$0);
}
            "#,
            r#"
struct FooDesc { pub bar: bool }

fn create_foo(foo_desc: &FooDesc) -> () { () }

fn baz() {
    let foo = create_foo(&FooDesc { bar: ${1:()} }$0);
}
            "#,
        )
    }

    #[test]
    fn literal_struct_impl_self_completion() {
        check_edit(
            "Self {…}",
            r#"
struct Foo {
    bar: u64,
}

impl Foo {
    fn new() -> Foo {
        Self$0
    }
}
            "#,
            r#"
struct Foo {
    bar: u64,
}

impl Foo {
    fn new() -> Foo {
        Self { bar: ${1:()} }$0
    }
}
            "#,
        );

        check_edit(
            "Self(…)",
            r#"
mod submod {
    pub struct Foo(pub u64);
}

impl submod::Foo {
    fn new() -> submod::Foo {
        Self$0
    }
}
            "#,
            r#"
mod submod {
    pub struct Foo(pub u64);
}

impl submod::Foo {
    fn new() -> submod::Foo {
        Self(${1:()})$0
    }
}
            "#,
        )
    }

    #[test]
    fn literal_struct_completion_from_sub_modules() {
        check_edit(
            "submod::Struct {…}",
            r#"
mod submod {
    pub struct Struct {
        pub a: u64,
    }
}

fn f() -> submod::Struct {
    Stru$0
}
            "#,
            r#"
mod submod {
    pub struct Struct {
        pub a: u64,
    }
}

fn f() -> submod::Struct {
    submod::Struct { a: ${1:()} }$0
}
            "#,
        )
    }

    #[test]
    fn literal_struct_complexion_module() {
        check_edit(
            "FooDesc {…}",
            r#"
mod _69latrick {
    pub struct FooDesc { pub six: bool, pub neuf: Vec<String>, pub bar: bool }
    pub fn create_foo(foo_desc: &FooDesc) -> () { () }
}

fn baz() {
    use _69latrick::*;

    let foo = create_foo(&$0);
}
            "#,
            r#"
mod _69latrick {
    pub struct FooDesc { pub six: bool, pub neuf: Vec<String>, pub bar: bool }
    pub fn create_foo(foo_desc: &FooDesc) -> () { () }
}

fn baz() {
    use _69latrick::*;

    let foo = create_foo(&FooDesc { six: ${1:()}, neuf: ${2:()}, bar: ${3:()} }$0);
}
            "#,
        );
    }

    #[test]
    fn default_completion_edit() {
        check_edit(
            "..Default::default()",
            r#"
//- minicore: default
struct Struct { foo: u32, bar: usize }

impl Default for Struct {
    fn default() -> Self {}
}

fn foo() {
    let other = Struct {
        foo: 5,
        .$0
    };
}
"#,
            r#"
struct Struct { foo: u32, bar: usize }

impl Default for Struct {
    fn default() -> Self {}
}

fn foo() {
    let other = Struct {
        foo: 5,
        ..Default::default()
    };
}
"#,
        );
        check_edit(
            "..Default::default()",
            r#"
//- minicore: default
struct Struct { foo: u32, bar: usize }

impl Default for Struct {
    fn default() -> Self {}
}

fn foo() {
    let other = Struct {
        foo: 5,
        $0
    };
}
"#,
            r#"
struct Struct { foo: u32, bar: usize }

impl Default for Struct {
    fn default() -> Self {}
}

fn foo() {
    let other = Struct {
        foo: 5,
        ..Default::default()
    };
}
"#,
        );
        check_edit(
            "..Default::default()",
            r#"
//- minicore: default
struct Struct { foo: u32, bar: usize }

impl Default for Struct {
    fn default() -> Self {}
}

fn foo() {
    let other = Struct {
        foo: 5,
        ..$0
    };
}
"#,
            r#"
struct Struct { foo: u32, bar: usize }

impl Default for Struct {
    fn default() -> Self {}
}

fn foo() {
    let other = Struct {
        foo: 5,
        ..Default::default()
    };
}
"#,
        );
    }
}
