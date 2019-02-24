use hir::{Ty, AdtDef, Docs};

use crate::completion::{CompletionContext, Completions, CompletionItem, CompletionItemKind};
use crate::completion::completion_item::CompletionKind;

/// Complete dot accesses, i.e. fields or methods (currently only fields).
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
    use crate::completion::*;
    use crate::completion::completion_item::check_completion;

    fn check_ref_completion(name: &str, code: &str) {
        check_completion(name, code, CompletionKind::Reference);
    }

    #[test]
    fn test_struct_literal_field() {
        check_ref_completion(
            "test_struct_literal_field",
            r"
            struct A { the_field: u32 }
            fn foo() {
               A { the<|> }
            }
            ",
        );
    }
}
