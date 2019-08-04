use hir::{AdtDef, Ty, TypeCtor};

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
        complete_fields(acc, ctx, receiver_ty.clone());
    }
    complete_methods(acc, ctx, receiver_ty.clone());

    // Suggest .await syntax for types that implement Future trait
    if ctx.analyzer.impls_future(ctx.db, receiver_ty) {
        CompletionItem::new(CompletionKind::Keyword, ctx.source_range(), "await")
            .detail("expr.await")
            .insert_text("await")
            .add_to(acc);
    }
}

fn complete_fields(acc: &mut Completions, ctx: &CompletionContext, receiver: Ty) {
    for receiver in ctx.analyzer.autoderef(ctx.db, receiver) {
        if let Ty::Apply(a_ty) = receiver {
            match a_ty.ctor {
                TypeCtor::Adt(AdtDef::Struct(s)) => {
                    for field in s.fields(ctx.db) {
                        acc.add_field(ctx, field, &a_ty.parameters);
                    }
                }
                // FIXME unions
                TypeCtor::Tuple { .. } => {
                    for (i, ty) in a_ty.parameters.iter().enumerate() {
                        acc.add_pos_field(ctx, i, ty);
                    }
                }
                _ => {}
            }
        };
    }
}

fn complete_methods(acc: &mut Completions, ctx: &CompletionContext, receiver: Ty) {
    let mut seen_methods = FxHashSet::default();
    ctx.analyzer.iterate_method_candidates(ctx.db, receiver, None, |_ty, func| {
        let data = func.data(ctx.db);
        if data.has_self_param() && seen_methods.insert(data.name().clone()) {
            acc.add_function(ctx, func);
        }
        None::<()>
    });
}

#[cfg(test)]
mod tests {
    use crate::completion::{do_completion, CompletionItem, CompletionKind};
    use insta::assert_debug_snapshot_matches;

    fn do_ref_completion(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Reference)
    }

    #[test]
    fn test_struct_field_completion() {
        assert_debug_snapshot_matches!(
        do_ref_completion(
                r"
                struct A { the_field: u32 }
                fn foo(a: A) {
                a.<|>
                }
                ",
        ),
            @r###"
       ⋮[
       ⋮    CompletionItem {
       ⋮        label: "the_field",
       ⋮        source_range: [94; 94),
       ⋮        delete: [94; 94),
       ⋮        insert: "the_field",
       ⋮        kind: Field,
       ⋮        detail: "u32",
       ⋮    },
       ⋮]
        "###
        );
    }

    #[test]
    fn test_struct_field_completion_self() {
        assert_debug_snapshot_matches!(
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
       ⋮[
       ⋮    CompletionItem {
       ⋮        label: "foo",
       ⋮        source_range: [187; 187),
       ⋮        delete: [187; 187),
       ⋮        insert: "foo()$0",
       ⋮        kind: Method,
       ⋮        detail: "fn foo(self)",
       ⋮    },
       ⋮    CompletionItem {
       ⋮        label: "the_field",
       ⋮        source_range: [187; 187),
       ⋮        delete: [187; 187),
       ⋮        insert: "the_field",
       ⋮        kind: Field,
       ⋮        detail: "(u32,)",
       ⋮        documentation: Documentation(
       ⋮            "This is the_field",
       ⋮        ),
       ⋮    },
       ⋮]
        "###
        );
    }

    #[test]
    fn test_struct_field_completion_autoderef() {
        assert_debug_snapshot_matches!(
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
       ⋮[
       ⋮    CompletionItem {
       ⋮        label: "foo",
       ⋮        source_range: [126; 126),
       ⋮        delete: [126; 126),
       ⋮        insert: "foo()$0",
       ⋮        kind: Method,
       ⋮        detail: "fn foo(&self)",
       ⋮    },
       ⋮    CompletionItem {
       ⋮        label: "the_field",
       ⋮        source_range: [126; 126),
       ⋮        delete: [126; 126),
       ⋮        insert: "the_field",
       ⋮        kind: Field,
       ⋮        detail: "(u32, i32)",
       ⋮    },
       ⋮]
        "###
        );
    }

    #[test]
    fn test_no_struct_field_completion_for_method_call() {
        assert_debug_snapshot_matches!(
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
    fn test_method_completion() {
        assert_debug_snapshot_matches!(
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
       ⋮[
       ⋮    CompletionItem {
       ⋮        label: "the_method",
       ⋮        source_range: [144; 144),
       ⋮        delete: [144; 144),
       ⋮        insert: "the_method()$0",
       ⋮        kind: Method,
       ⋮        detail: "fn the_method(&self)",
       ⋮    },
       ⋮]
        "###
        );
    }

    #[test]
    fn test_trait_method_completion() {
        assert_debug_snapshot_matches!(
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
       ⋮[
       ⋮    CompletionItem {
       ⋮        label: "the_method",
       ⋮        source_range: [151; 151),
       ⋮        delete: [151; 151),
       ⋮        insert: "the_method()$0",
       ⋮        kind: Method,
       ⋮        detail: "fn the_method(&self)",
       ⋮    },
       ⋮]
        "###
        );
    }

    #[test]
    fn test_trait_method_completion_deduplicated() {
        assert_debug_snapshot_matches!(
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
       ⋮[
       ⋮    CompletionItem {
       ⋮        label: "the_method",
       ⋮        source_range: [155; 155),
       ⋮        delete: [155; 155),
       ⋮        insert: "the_method()$0",
       ⋮        kind: Method,
       ⋮        detail: "fn the_method(&self)",
       ⋮    },
       ⋮]
        "###
        );
    }

    #[test]
    fn test_no_non_self_method() {
        assert_debug_snapshot_matches!(
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
        assert_debug_snapshot_matches!(
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
       ⋮[
       ⋮    CompletionItem {
       ⋮        label: "the_method",
       ⋮        source_range: [249; 249),
       ⋮        delete: [249; 249),
       ⋮        insert: "the_method()$0",
       ⋮        kind: Method,
       ⋮        detail: "fn the_method(&self)",
       ⋮    },
       ⋮]
        "###
        );
    }

    #[test]
    fn test_tuple_field_completion() {
        assert_debug_snapshot_matches!(
        do_ref_completion(
            r"
            fn foo() {
               let b = (0, 3.14);
               b.<|>
            }
            ",
        ),
        @r###"
       ⋮[
       ⋮    CompletionItem {
       ⋮        label: "0",
       ⋮        source_range: [75; 75),
       ⋮        delete: [75; 75),
       ⋮        insert: "0",
       ⋮        kind: Field,
       ⋮        detail: "i32",
       ⋮    },
       ⋮    CompletionItem {
       ⋮        label: "1",
       ⋮        source_range: [75; 75),
       ⋮        delete: [75; 75),
       ⋮        insert: "1",
       ⋮        kind: Field,
       ⋮        detail: "f64",
       ⋮    },
       ⋮]
        "###
        );
    }

    #[test]
    fn test_tuple_field_inference() {
        assert_debug_snapshot_matches!(
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
       ⋮[
       ⋮    CompletionItem {
       ⋮        label: "blah",
       ⋮        source_range: [299; 300),
       ⋮        delete: [299; 300),
       ⋮        insert: "blah()$0",
       ⋮        kind: Method,
       ⋮        detail: "pub fn blah(&self)",
       ⋮    },
       ⋮]
        "###
        );
    }

    #[test]
    fn test_completion_works_in_consts() {
        assert_debug_snapshot_matches!(
        do_ref_completion(
            r"
            struct A { the_field: u32 }
            const X: u32 = {
                A { the_field: 92 }.<|>
            };
            ",
        ),
        @r###"
       ⋮[
       ⋮    CompletionItem {
       ⋮        label: "the_field",
       ⋮        source_range: [106; 106),
       ⋮        delete: [106; 106),
       ⋮        insert: "the_field",
       ⋮        kind: Field,
       ⋮        detail: "u32",
       ⋮    },
       ⋮]
        "###
        );
    }

    #[test]
    fn test_completion_await_impls_future() {
        assert_debug_snapshot_matches!(
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
       ⋮[
       ⋮    CompletionItem {
       ⋮        label: "await",
       ⋮        source_range: [74; 74),
       ⋮        delete: [74; 74),
       ⋮        insert: "await",
       ⋮        detail: "expr.await",
       ⋮    },
       ⋮]
        "###
        )
    }
}
