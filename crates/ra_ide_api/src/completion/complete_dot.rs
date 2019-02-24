use hir::{Ty, AdtDef};

use crate::completion::{CompletionContext, Completions, CompletionKind};

/// Complete dot accesses, i.e. fields or methods (currently only fields).
pub(super) fn complete_dot(acc: &mut Completions, ctx: &CompletionContext) {
    let (function, receiver) = match (&ctx.function, ctx.dot_receiver) {
        (Some(function), Some(receiver)) => (function, receiver),
        _ => return,
    };
    let infer_result = function.infer(ctx.db);
    let syntax_mapping = function.body_syntax_mapping(ctx.db);
    let expr = match syntax_mapping.node_expr(receiver) {
        Some(expr) => expr,
        None => return,
    };
    let receiver_ty = infer_result[expr].clone();
    if !ctx.is_call {
        complete_fields(acc, ctx, receiver_ty.clone());
    }
    complete_methods(acc, ctx, receiver_ty);
}

fn complete_fields(acc: &mut Completions, ctx: &CompletionContext, receiver: Ty) {
    for receiver in receiver.autoderef(ctx.db) {
        match receiver {
            Ty::Adt { def_id, ref substs, .. } => {
                match def_id {
                    AdtDef::Struct(s) => {
                        for field in s.fields(ctx.db) {
                            acc.add_field(CompletionKind::Reference, ctx, field, substs);
                        }
                    }

                    // TODO unions
                    AdtDef::Enum(_) => (),
                }
            }
            Ty::Tuple(fields) => {
                for (i, ty) in fields.iter().enumerate() {
                    acc.add_pos_field(CompletionKind::Reference, ctx, i, ty);
                }
            }
            _ => {}
        };
    }
}

fn complete_methods(acc: &mut Completions, ctx: &CompletionContext, receiver: Ty) {
    receiver.iterate_methods(ctx.db, |_ty, func| {
        let sig = func.signature(ctx.db);
        if sig.has_self_param() {
            acc.add_function(CompletionKind::Reference, ctx, func);
        }
        None::<()>
    });
}

#[cfg(test)]
mod tests {
    use crate::completion::*;
    use crate::completion::completion_item::check_completion;

    fn check_ref_completion(name: &str, code: &str) {
        check_completion(name, code, CompletionKind::Reference);
    }

    #[test]
    fn test_struct_field_completion() {
        check_ref_completion(
            "struct_field_completion",
            r"
            struct A { the_field: u32 }
            fn foo(a: A) {
               a.<|>
            }
            ",
        );
    }

    #[test]
    fn test_struct_field_completion_self() {
        check_ref_completion(
            "struct_field_completion_self",
            r"
            struct A {
                /// This is the_field
                the_field: (u32,)
            }
            impl A {
                fn foo(self) {
                    self.<|>
                }
            }
            ",
        );
    }

    #[test]
    fn test_struct_field_completion_autoderef() {
        check_ref_completion(
            "struct_field_completion_autoderef",
            r"
            struct A { the_field: (u32, i32) }
            impl A {
                fn foo(&self) {
                    self.<|>
                }
            }
            ",
        );
    }

    #[test]
    fn test_no_struct_field_completion_for_method_call() {
        check_ref_completion(
            "no_struct_field_completion_for_method_call",
            r"
            struct A { the_field: u32 }
            fn foo(a: A) {
               a.<|>()
            }
            ",
        );
    }

    #[test]
    fn test_method_completion() {
        check_ref_completion(
            "method_completion",
            r"
            struct A {}
            impl A {
                fn the_method(&self) {}
            }
            fn foo(a: A) {
               a.<|>
            }
            ",
        );
    }

    #[test]
    fn test_no_non_self_method() {
        check_ref_completion(
            "no_non_self_method",
            r"
            struct A {}
            impl A {
                fn the_method() {}
            }
            fn foo(a: A) {
               a.<|>
            }
            ",
        );
    }

    #[test]
    fn test_method_attr_filtering() {
        check_ref_completion(
            "method_attr_filtering",
            r"
            struct A {}
            impl A {
                #[inline]
                fn the_method(&self) {
                    let x = 1;
                    let y = 2;
                }
            }
            fn foo(a: A) {
               a.<|>
            }
            ",
        );
    }

    #[test]
    fn test_tuple_field_completion() {
        check_ref_completion(
            "tuple_field_completion",
            r"
            fn foo() {
               let b = (0, 3.14);
               b.<|>
            }
            ",
        );
    }
}
