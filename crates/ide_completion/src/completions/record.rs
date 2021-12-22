//! Complete fields in record literals and patterns.
use ide_db::SymbolKind;
use syntax::{ast::Expr, T};

use crate::{
    patterns::ImmediateLocation, CompletionContext, CompletionItem, CompletionItemKind, Completions,
};

pub(crate) fn complete_record(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    let missing_fields = match &ctx.completion_location {
        Some(
            ImmediateLocation::RecordExpr(record_expr)
            | ImmediateLocation::RecordExprUpdate(record_expr),
        ) => {
            let ty = ctx.sema.type_of_expr(&Expr::RecordExpr(record_expr.clone()));
            let default_trait = ctx.famous_defs().core_default_Default();
            let impl_default_trait = default_trait.zip(ty).map_or(false, |(default_trait, ty)| {
                ty.original.impls_trait(ctx.db, default_trait, &[])
            });

            let missing_fields = ctx.sema.record_literal_missing_fields(record_expr);
            if impl_default_trait && !missing_fields.is_empty() && ctx.path_qual().is_none() {
                let completion_text = "..Default::default()";
                let mut item =
                    CompletionItem::new(SymbolKind::Field, ctx.source_range(), completion_text);
                let completion_text =
                    completion_text.strip_prefix(ctx.token.text()).unwrap_or(completion_text);
                item.insert_text(completion_text);
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
        Some(ImmediateLocation::RecordPat(record_pat)) => {
            ctx.sema.record_pattern_missing_fields(record_pat)
        }
        _ => return None,
    };

    for (field, ty) in missing_fields {
        acc.add_field(ctx, None, field, &ty);
    }

    Some(())
}

pub(crate) fn complete_record_literal(
    acc: &mut Completions,
    ctx: &CompletionContext,
) -> Option<()> {
    if !ctx.expects_expression() {
        return None;
    }

    if let hir::Adt::Struct(strukt) = ctx.expected_type.as_ref()?.as_adt()? {
        let module =
            if let Some(module) = ctx.scope.module() { module } else { strukt.module(ctx.db) };

        let path = module.find_use_path(ctx.db, hir::ModuleDef::from(strukt));

        acc.add_struct_literal(ctx, strukt, path, None);
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
    fn literal_struct_completion_from_sub_modules() {
        check_edit(
            "Struct {…}",
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
