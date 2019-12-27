//! FIXME: write short doc here

use hir::{HasVisibility, Type};

use crate::completion::completion_item::CompletionKind;
use crate::{
    completion::{completion_context::CompletionContext, completion_item::Completions},
    CompletionItem,
};
use rustc_hash::FxHashSet;

/// Complete dot accesses, i.e. fields or methods (and .await syntax).
pub(super) fn complete_dot(acc: &mut Completions, ctx: &CompletionContext) {
    let dot_receiver = match &ctx.dot_receiver {
        Some(expr) => expr,
        _ => return,
    };

    let receiver_ty = match ctx.analyzer.type_of(ctx.db, &dot_receiver) {
        Some(ty) => ty,
        _ => return,
    };

    if !ctx.is_call {
        complete_fields(acc, ctx, &receiver_ty);
    }
    complete_methods(acc, ctx, &receiver_ty);

    // Suggest .await syntax for types that implement Future trait
    if ctx.analyzer.impls_future(ctx.db, receiver_ty) {
        CompletionItem::new(CompletionKind::Keyword, ctx.source_range(), "await")
            .detail("expr.await")
            .insert_text("await")
            .add_to(acc);
    }
}

fn complete_fields(acc: &mut Completions, ctx: &CompletionContext, receiver: &Type) {
    for receiver in receiver.autoderef(ctx.db) {
        for (field, ty) in receiver.fields(ctx.db) {
            if ctx.module.map_or(false, |m| !field.is_visible_from(ctx.db, m)) {
                // Skip private field. FIXME: If the definition location of the
                // field is editable, we should show the completion
                continue;
            }
            acc.add_field(ctx, field, &ty);
        }
        for (i, ty) in receiver.tuple_fields(ctx.db).into_iter().enumerate() {
            // FIXME: Handle visibility
            acc.add_tuple_field(ctx, i, &ty);
        }
    }
}

fn complete_methods(acc: &mut Completions, ctx: &CompletionContext, receiver: &Type) {
    let mut seen_methods = FxHashSet::default();
    ctx.analyzer.iterate_method_candidates(ctx.db, receiver, None, |_ty, func| {
        if func.has_self_param(ctx.db) && seen_methods.insert(func.name(ctx.db)) {
            acc.add_function(ctx, func);
        }
        None::<()>
    });
}

#[cfg(test)]
mod tests {
    use crate::completion::{do_completion, CompletionItem, CompletionKind};
    use insta::assert_debug_snapshot;

    fn do_ref_completion(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Reference)
    }

    #[test]
    fn test_struct_field_completion() {
        assert_debug_snapshot!(
        do_ref_completion(
                r"
                struct A { the_field: u32 }
                fn foo(a: A) {
                a.<|>
                }
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "the_field",
                source_range: [94; 94),
                delete: [94; 94),
                insert: "the_field",
                kind: Field,
                detail: "u32",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_struct_field_completion_self() {
        assert_debug_snapshot!(
        do_ref_completion(
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
        ),
        @r###"
        [
            CompletionItem {
                label: "foo()",
                source_range: [187; 187),
                delete: [187; 187),
                insert: "foo()$0",
                kind: Method,
                lookup: "foo",
                detail: "fn foo(self)",
            },
            CompletionItem {
                label: "the_field",
                source_range: [187; 187),
                delete: [187; 187),
                insert: "the_field",
                kind: Field,
                detail: "(u32,)",
                documentation: Documentation(
                    "This is the_field",
                ),
            },
        ]
        "###
        );
    }

    #[test]
    fn test_struct_field_completion_autoderef() {
        assert_debug_snapshot!(
        do_ref_completion(
            r"
            struct A { the_field: (u32, i32) }
            impl A {
                fn foo(&self) {
                    self.<|>
                }
            }
            ",
        ),
        @r###"
        [
            CompletionItem {
                label: "foo()",
                source_range: [126; 126),
                delete: [126; 126),
                insert: "foo()$0",
                kind: Method,
                lookup: "foo",
                detail: "fn foo(&self)",
            },
            CompletionItem {
                label: "the_field",
                source_range: [126; 126),
                delete: [126; 126),
                insert: "the_field",
                kind: Field,
                detail: "(u32, i32)",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_no_struct_field_completion_for_method_call() {
        assert_debug_snapshot!(
        do_ref_completion(
            r"
            struct A { the_field: u32 }
            fn foo(a: A) {
               a.<|>()
            }
            ",
        ),
        @"[]"
        );
    }

    #[test]
    fn test_struct_field_visibility_private() {
        assert_debug_snapshot!(
            do_ref_completion(
                r"
            mod inner {
                struct A {
                    private_field: u32,
                    pub pub_field: u32,
                    pub(crate) crate_field: u32,
                    pub(super) super_field: u32,
                }
            }
            fn foo(a: inner::A) {
               a.<|>
            }
            ",
            ),
            @r###"
        [
            CompletionItem {
                label: "crate_field",
                source_range: [313; 313),
                delete: [313; 313),
                insert: "crate_field",
                kind: Field,
                detail: "u32",
            },
            CompletionItem {
                label: "pub_field",
                source_range: [313; 313),
                delete: [313; 313),
                insert: "pub_field",
                kind: Field,
                detail: "u32",
            },
            CompletionItem {
                label: "super_field",
                source_range: [313; 313),
                delete: [313; 313),
                insert: "super_field",
                kind: Field,
                detail: "u32",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_method_completion() {
        assert_debug_snapshot!(
        do_ref_completion(
            r"
            struct A {}
            impl A {
                fn the_method(&self) {}
            }
            fn foo(a: A) {
               a.<|>
            }
            ",
        ),
        @r###"
        [
            CompletionItem {
                label: "the_method()",
                source_range: [144; 144),
                delete: [144; 144),
                insert: "the_method()$0",
                kind: Method,
                lookup: "the_method",
                detail: "fn the_method(&self)",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_method_completion_only_fitting_impls() {
        assert_debug_snapshot!(
            do_ref_completion(
                r"
            struct A<T> {}
            impl A<u32> {
                fn the_method(&self) {}
            }
            impl A<i32> {
                fn the_other_method(&self) {}
            }
            fn foo(a: A<u32>) {
               a.<|>
            }
            ",
            ),
            @r###"
        [
            CompletionItem {
                label: "the_method()",
                source_range: [243; 243),
                delete: [243; 243),
                insert: "the_method()$0",
                kind: Method,
                lookup: "the_method",
                detail: "fn the_method(&self)",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_trait_method_completion() {
        assert_debug_snapshot!(
            do_ref_completion(
                r"
            struct A {}
            trait Trait { fn the_method(&self); }
            impl Trait for A {}
            fn foo(a: A) {
               a.<|>
            }
            ",
            ),
            @r###"
        [
            CompletionItem {
                label: "the_method()",
                source_range: [151; 151),
                delete: [151; 151),
                insert: "the_method()$0",
                kind: Method,
                lookup: "the_method",
                detail: "fn the_method(&self)",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_trait_method_completion_deduplicated() {
        assert_debug_snapshot!(
            do_ref_completion(
                r"
            struct A {}
            trait Trait { fn the_method(&self); }
            impl<T> Trait for T {}
            fn foo(a: &A) {
               a.<|>
            }
            ",
            ),
            @r###"
        [
            CompletionItem {
                label: "the_method()",
                source_range: [155; 155),
                delete: [155; 155),
                insert: "the_method()$0",
                kind: Method,
                lookup: "the_method",
                detail: "fn the_method(&self)",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_no_non_self_method() {
        assert_debug_snapshot!(
        do_ref_completion(
            r"
            struct A {}
            impl A {
                fn the_method() {}
            }
            fn foo(a: A) {
               a.<|>
            }
            ",
        ),
        @"[]"
        );
    }

    #[test]
    fn test_method_attr_filtering() {
        assert_debug_snapshot!(
        do_ref_completion(
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
        ),
        @r###"
        [
            CompletionItem {
                label: "the_method()",
                source_range: [249; 249),
                delete: [249; 249),
                insert: "the_method()$0",
                kind: Method,
                lookup: "the_method",
                detail: "fn the_method(&self)",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_tuple_field_completion() {
        assert_debug_snapshot!(
        do_ref_completion(
            r"
            fn foo() {
               let b = (0, 3.14);
               b.<|>
            }
            ",
        ),
        @r###"
        [
            CompletionItem {
                label: "0",
                source_range: [75; 75),
                delete: [75; 75),
                insert: "0",
                kind: Field,
                detail: "i32",
            },
            CompletionItem {
                label: "1",
                source_range: [75; 75),
                delete: [75; 75),
                insert: "1",
                kind: Field,
                detail: "f64",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_tuple_field_inference() {
        assert_debug_snapshot!(
        do_ref_completion(
            r"
            pub struct S;
            impl S {
                pub fn blah(&self) {}
            }

            struct T(S);

            impl T {
                fn foo(&self) {
                    // FIXME: This doesn't work without the trailing `a` as `0.` is a float
                    self.0.a<|>
                }
            }
            ",
        ),
        @r###"
        [
            CompletionItem {
                label: "blah()",
                source_range: [299; 300),
                delete: [299; 300),
                insert: "blah()$0",
                kind: Method,
                lookup: "blah",
                detail: "pub fn blah(&self)",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_completion_works_in_consts() {
        assert_debug_snapshot!(
        do_ref_completion(
            r"
            struct A { the_field: u32 }
            const X: u32 = {
                A { the_field: 92 }.<|>
            };
            ",
        ),
        @r###"
        [
            CompletionItem {
                label: "the_field",
                source_range: [106; 106),
                delete: [106; 106),
                insert: "the_field",
                kind: Field,
                detail: "u32",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_completion_await_impls_future() {
        assert_debug_snapshot!(
        do_completion(
            r###"
            //- /main.rs
            use std::future::*;
            struct A {}
            impl Future for A {}
            fn foo(a: A) {
                a.<|>
            }

            //- /std/lib.rs
            pub mod future {
                pub trait Future {}
            }
            "###, CompletionKind::Keyword),
        @r###"
        [
            CompletionItem {
                label: "await",
                source_range: [74; 74),
                delete: [74; 74),
                insert: "await",
                detail: "expr.await",
            },
        ]
        "###
        )
    }
}
