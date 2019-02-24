use hir::{Ty, AdtDef, Docs};

use crate::completion::{CompletionContext, Completions, CompletionItem, CompletionItemKind};
use crate::completion::completion_item::CompletionKind;

/// Complete fields in fields literals.
pub(super) fn complete_struct_literal(acc: &mut Completions, ctx: &CompletionContext) {
    let (function, struct_lit) = match (&ctx.function, ctx.struct_lit_syntax) {
        (Some(function), Some(struct_lit)) => (function, struct_lit),
        _ => return,
    };
    let infer_result = function.infer(ctx.db);
    let syntax_mapping = function.body_syntax_mapping(ctx.db);
    let expr = match syntax_mapping.node_expr(struct_lit.into()) {
        Some(expr) => expr,
        None => return,
    };
    let ty = infer_result[expr].clone();
    let (adt, substs) = match ty {
        Ty::Adt { def_id, ref substs, .. } => (def_id, substs),
        _ => return,
    };
    match adt {
        AdtDef::Struct(s) => {
            for field in s.fields(ctx.db) {
                CompletionItem::new(
                    CompletionKind::Reference,
                    ctx.source_range(),
                    field.name(ctx.db).to_string(),
                )
                .kind(CompletionItemKind::Field)
                .detail(field.ty(ctx.db).subst(substs).to_string())
                .set_documentation(field.docs(ctx.db))
                .add_to(acc);
            }
        }

        // TODO unions
        AdtDef::Enum(_) => (),
    };
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot_matches;
    use crate::completion::{CompletionItem, CompletionKind};

    fn complete(code: &str) -> Vec<CompletionItem> {
        crate::completion::completion_item::do_completion(code, CompletionKind::Reference)
    }

    #[test]
    fn test_struct_literal_field() {
        let completions = complete(
            r"
            struct A { the_field: u32 }
            fn foo() {
               A { the<|> }
            }
            ",
        );
        assert_debug_snapshot_matches!(completions, @r###"[
    CompletionItem {
        label: "the_field",
        source_range: [83; 86),
        delete: [83; 86),
        insert: "the_field",
        kind: Field,
        detail: "u32"
    }
]"###);
    }
}
