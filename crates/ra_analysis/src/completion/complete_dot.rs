use ra_syntax::ast::AstNode;
use hir::{Ty, Def};

use crate::Cancelable;
use crate::completion::{CompletionContext, Completions, CompletionKind, CompletionItem, CompletionItemKind};

/// Complete dot accesses, i.e. fields or methods (currently only fields).
pub(super) fn complete_dot(acc: &mut Completions, ctx: &CompletionContext) -> Cancelable<()> {
    let (function, receiver) = match (&ctx.function, ctx.dot_receiver) {
        (Some(function), Some(receiver)) => (function, receiver),
        _ => return Ok(()),
    };
    let infer_result = function.infer(ctx.db)?;
    let receiver_ty = if let Some(ty) = infer_result.type_of_node(receiver.syntax()) {
        ty
    } else {
        return Ok(());
    };
    if !ctx.is_method_call {
        complete_fields(acc, ctx, receiver_ty)?;
    }
    Ok(())
}

fn complete_fields(acc: &mut Completions, ctx: &CompletionContext, receiver: Ty) -> Cancelable<()> {
    // TODO: autoderef etc.
    match receiver {
        Ty::Adt { def_id, .. } => {
            match def_id.resolve(ctx.db)? {
                Def::Struct(s) => {
                    let variant_data = s.variant_data(ctx.db)?;
                    for field in variant_data.fields() {
                        CompletionItem::new(CompletionKind::Reference, field.name().to_string())
                            .kind(CompletionItemKind::Field)
                            .add_to(acc);
                    }
                }
                // TODO unions
                _ => {}
            }
        }
        Ty::Tuple(fields) => {
            for (i, _ty) in fields.iter().enumerate() {
                CompletionItem::new(CompletionKind::Reference, i.to_string())
                    .kind(CompletionItemKind::Field)
                    .add_to(acc);
            }
        }
        _ => {}
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::completion::*;

    fn check_ref_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Reference);
    }

    #[test]
    fn test_struct_field_completion() {
        check_ref_completion(
            r"
            struct A { the_field: u32 }
            fn foo(a: A) {
               a.<|>
            }
            ",
            r#"the_field"#,
        );
    }

    #[test]
    fn test_no_struct_field_completion_for_method_call() {
        check_ref_completion(
            r"
            struct A { the_field: u32 }
            fn foo(a: A) {
               a.<|>()
            }
            ",
            r#""#,
        );
    }
}
