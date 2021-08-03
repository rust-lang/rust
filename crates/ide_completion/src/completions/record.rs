//! Complete fields in record literals and patterns.
use ide_db::{helpers::FamousDefs, SymbolKind};
use syntax::ast::Expr;

use crate::{
    item::CompletionKind, patterns::ImmediateLocation, CompletionContext, CompletionItem,
    Completions,
};

pub(crate) fn complete_record(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    let missing_fields = match &ctx.completion_location {
        Some(ImmediateLocation::RecordExpr(record_expr)) => {
            let ty = ctx.sema.type_of_expr(&Expr::RecordExpr(record_expr.clone()));
            let default_trait = FamousDefs(&ctx.sema, ctx.krate).core_default_Default();
            let impl_default_trait = default_trait.zip(ty).map_or(false, |(default_trait, ty)| {
                ty.original.impls_trait(ctx.db, default_trait, &[])
            });

            let missing_fields = ctx.sema.record_literal_missing_fields(record_expr);
            if impl_default_trait && !missing_fields.is_empty() {
                let completion_text = "..Default::default()";
                let mut item = CompletionItem::new(
                    CompletionKind::Snippet,
                    ctx.source_range(),
                    completion_text,
                );
                let completion_text =
                    completion_text.strip_prefix(ctx.token.text()).unwrap_or(completion_text);
                item.insert_text(completion_text).kind(SymbolKind::Field);
                item.add_to(acc);
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

#[cfg(test)]
mod tests {
    use crate::tests::check_edit;

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
    }
}
