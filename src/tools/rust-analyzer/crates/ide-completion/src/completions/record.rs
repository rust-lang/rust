//! Complete fields in record literals and patterns.
use ide_db::SymbolKind;
use syntax::{
    SmolStr,
    ast::{self, Expr},
};

use crate::{
    CompletionContext, CompletionItem, CompletionItemKind, CompletionRelevance,
    CompletionRelevancePostfixMatch, Completions,
    context::{DotAccess, DotAccessExprCtx, DotAccessKind, PatternContext},
};

pub(crate) fn complete_record_pattern_fields(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    pattern_ctx: &PatternContext,
) {
    if let PatternContext { record_pat: Some(record_pat), .. } = pattern_ctx {
        let ty = ctx.sema.type_of_pat(&ast::Pat::RecordPat(record_pat.clone()));
        let missing_fields = match ty.as_ref().and_then(|t| t.original.as_adt()) {
            Some(hir::Adt::Union(un)) => {
                // ctx.sema.record_pat_missing_fields will always return
                // an empty Vec on a union literal. This is normally
                // reasonable, but here we'd like to present the full list
                // of fields if the literal is empty.
                let were_fields_specified =
                    record_pat.record_pat_field_list().and_then(|fl| fl.fields().next()).is_some();

                match were_fields_specified {
                    false => un
                        .fields(ctx.db)
                        .into_iter()
                        .map(|f| (f, f.ty(ctx.db).to_type(ctx.db)))
                        .collect(),
                    true => return,
                }
            }
            _ => ctx.sema.record_pattern_matched_fields(record_pat),
        };
        complete_fields(acc, ctx, missing_fields);
    }
}

pub(crate) fn complete_record_expr_fields(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    record_expr: &ast::RecordExpr,
    &dot_prefix: &bool,
) {
    let ty = ctx.sema.type_of_expr(&Expr::RecordExpr(record_expr.clone()));

    let missing_fields = match ty.as_ref().and_then(|t| t.original.as_adt()) {
        Some(hir::Adt::Union(un)) => {
            // ctx.sema.record_literal_missing_fields will always return
            // an empty Vec on a union literal. This is normally
            // reasonable, but here we'd like to present the full list
            // of fields if the literal is empty.
            let were_fields_specified =
                record_expr.record_expr_field_list().and_then(|fl| fl.fields().next()).is_some();

            match were_fields_specified {
                false => un
                    .fields(ctx.db)
                    .into_iter()
                    .map(|f| (f, f.ty(ctx.db).to_type(ctx.db)))
                    .collect(),
                true => return,
            }
        }
        _ => {
            let suggest_fields = ctx.sema.record_literal_matched_fields(record_expr);
            let update_exists = record_expr
                .record_expr_field_list()
                .is_some_and(|list| list.dotdot_token().is_some());

            if !suggest_fields.is_empty() && !update_exists {
                cov_mark::hit!(functional_update_field);
                add_default_update(acc, ctx, ty.as_ref());
            }
            if dot_prefix {
                cov_mark::hit!(functional_update_one_dot);
                let mut item = CompletionItem::new(
                    CompletionItemKind::Snippet,
                    ctx.source_range(),
                    SmolStr::new_static(".."),
                    ctx.edition,
                );
                item.insert_text(".");
                item.add_to(acc, ctx.db);
                return;
            }
            suggest_fields
        }
    };
    complete_fields(acc, ctx, missing_fields);
}

pub(crate) fn add_default_update(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    ty: Option<&hir::TypeInfo<'_>>,
) {
    let default_trait = ctx.famous_defs().core_default_Default();
    let impls_default_trait = default_trait
        .zip(ty)
        .is_some_and(|(default_trait, ty)| ty.original.impls_trait(ctx.db, default_trait, &[]));
    if impls_default_trait {
        // FIXME: This should make use of scope_def like completions so we get all the other goodies
        // that is we should handle this like actually completing the default function
        let completion_text = "..Default::default()";
        let mut item = CompletionItem::new(
            SymbolKind::Field,
            ctx.source_range(),
            SmolStr::new_static(completion_text),
            ctx.edition,
        );
        let completion_text =
            completion_text.strip_prefix(ctx.token.text()).unwrap_or(completion_text);
        item.insert_text(completion_text).set_relevance(CompletionRelevance {
            postfix_match: Some(CompletionRelevancePostfixMatch::Exact),
            ..Default::default()
        });
        item.add_to(acc, ctx.db);
    }
}

fn complete_fields(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    missing_fields: Vec<(hir::Field, hir::Type<'_>)>,
) {
    for (field, ty) in missing_fields {
        // This should call something else, we shouldn't be synthesizing a DotAccess here
        acc.add_field(
            ctx,
            &DotAccess {
                receiver: None,
                receiver_ty: None,
                kind: DotAccessKind::Field { receiver_is_ambiguous_float_literal: false },
                ctx: DotAccessExprCtx { in_block_expr: false, in_breakable: None },
            },
            None,
            field,
            &ty,
        );
    }
}

#[cfg(test)]
mod tests {
    use ide_db::SnippetCap;

    use crate::{
        CompletionConfig,
        tests::{TEST_CONFIG, check_edit, check_edit_with_config},
    };

    #[test]
    fn literal_struct_completion_edit() {
        check_edit(
            "FooDesc{}",
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
    fn literal_struct_completion_shorthand() {
        check_edit(
            "FooDesc{}",
            r#"
struct FooDesc { pub bar: bool, n: i32 }

fn create_foo(foo_desc: &FooDesc) -> () { () }

fn baz() {
    let bar = true;
    let foo = create_foo(&$0);
}
            "#,
            r#"
struct FooDesc { pub bar: bool, n: i32 }

fn create_foo(foo_desc: &FooDesc) -> () { () }

fn baz() {
    let bar = true;
    let foo = create_foo(&FooDesc { bar$1, n: ${2:()} }$0);
}
            "#,
        )
    }

    #[test]
    fn enum_variant_no_snippets() {
        let conf = CompletionConfig { snippet_cap: SnippetCap::new(false), ..TEST_CONFIG };
        // tuple variant
        check_edit_with_config(
            conf.clone(),
            "Variant()",
            r#"
enum Enum {
    Variant(usize),
}

impl Enum {
    fn new(u: usize) -> Self {
        Self::Va$0
    }
}
"#,
            r#"
enum Enum {
    Variant(usize),
}

impl Enum {
    fn new(u: usize) -> Self {
        Self::Variant
    }
}
"#,
        );

        // record variant
        check_edit_with_config(
            conf,
            "Variant{}",
            r#"
enum Enum {
    Variant{u: usize},
}

impl Enum {
    fn new(u: usize) -> Self {
        Self::Va$0
    }
}
"#,
            r#"
enum Enum {
    Variant{u: usize},
}

impl Enum {
    fn new(u: usize) -> Self {
        Self::Variant
    }
}
"#,
        )
    }

    #[test]
    fn literal_struct_impl_self_completion() {
        check_edit(
            "Self{}",
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
            "Self()",
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
            "submod::Struct{}",
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
            "FooDesc{}",
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

    #[test]
    fn callable_field_struct_init() {
        check_edit(
            "field",
            r#"
struct S {
    field: fn(),
}

fn main() {
    S {fi$0
}
"#,
            r#"
struct S {
    field: fn(),
}

fn main() {
    S {field
}
"#,
        );
    }
}
