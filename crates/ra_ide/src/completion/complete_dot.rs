//! FIXME: write short doc here

use hir::{HasVisibility, Type};

use crate::{
    completion::{
        completion_context::CompletionContext,
        completion_item::{CompletionKind, Completions},
    },
    CompletionItem,
};
use rustc_hash::FxHashSet;

/// Complete dot accesses, i.e. fields or methods (and .await syntax).
pub(super) fn complete_dot(acc: &mut Completions, ctx: &CompletionContext) {
    let dot_receiver = match &ctx.dot_receiver {
        Some(expr) => expr,
        _ => return,
    };

    let receiver_ty = match ctx.sema.type_of_expr(&dot_receiver) {
        Some(ty) => ty,
        _ => return,
    };

    if !ctx.is_call {
        complete_fields(acc, ctx, &receiver_ty);
    }
    complete_methods(acc, ctx, &receiver_ty);

    // Suggest .await syntax for types that implement Future trait
    if receiver_ty.impls_future(ctx.db) {
        CompletionItem::new(CompletionKind::Keyword, ctx.source_range(), "await")
            .detail("expr.await")
            .insert_text("await")
            .add_to(acc);
    }
}

fn complete_fields(acc: &mut Completions, ctx: &CompletionContext, receiver: &Type) {
    for receiver in receiver.autoderef(ctx.db) {
        for (field, ty) in receiver.fields(ctx.db) {
            if ctx.scope().module().map_or(false, |m| !field.is_visible_from(ctx.db, m)) {
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
    if let Some(krate) = ctx.krate {
        let mut seen_methods = FxHashSet::default();
        let traits_in_scope = ctx.scope().traits_in_scope();
        receiver.iterate_method_candidates(ctx.db, krate, &traits_in_scope, None, |_ty, func| {
            if func.has_self_param(ctx.db)
                && ctx.scope().module().map_or(true, |m| func.is_visible_from(ctx.db, m))
                && seen_methods.insert(func.name(ctx.db))
            {
                acc.add_function(ctx, func, None);
            }
            None::<()>
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::completion::{test_utils::do_completion, CompletionItem, CompletionKind};
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
                source_range: 45..45,
                delete: 45..45,
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
                source_range: 102..102,
                delete: 102..102,
                insert: "foo()$0",
                kind: Method,
                lookup: "foo",
                detail: "fn foo(self)",
            },
            CompletionItem {
                label: "the_field",
                source_range: 102..102,
                delete: 102..102,
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
                source_range: 77..77,
                delete: 77..77,
                insert: "foo()$0",
                kind: Method,
                lookup: "foo",
                detail: "fn foo(&self)",
            },
            CompletionItem {
                label: "the_field",
                source_range: 77..77,
                delete: 77..77,
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
                source_range: 192..192,
                delete: 192..192,
                insert: "crate_field",
                kind: Field,
                detail: "u32",
            },
            CompletionItem {
                label: "pub_field",
                source_range: 192..192,
                delete: 192..192,
                insert: "pub_field",
                kind: Field,
                detail: "u32",
            },
            CompletionItem {
                label: "super_field",
                source_range: 192..192,
                delete: 192..192,
                insert: "super_field",
                kind: Field,
                detail: "u32",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_union_field_completion() {
        assert_debug_snapshot!(
            do_ref_completion(
                r"
            union Un {
                field: u8,
                other: u16,
            }

            fn foo(u: Un) {
                u.<|>
            }
            ",
            ),
            @r###"
        [
            CompletionItem {
                label: "field",
                source_range: 67..67,
                delete: 67..67,
                insert: "field",
                kind: Field,
                detail: "u8",
            },
            CompletionItem {
                label: "other",
                source_range: 67..67,
                delete: 67..67,
                insert: "other",
                kind: Field,
                detail: "u16",
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
                source_range: 71..71,
                delete: 71..71,
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
                source_range: 134..134,
                delete: 134..134,
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
    fn test_method_completion_private() {
        assert_debug_snapshot!(
            do_ref_completion(
                r"
            struct A {}
            mod m {
                impl super::A {
                    fn private_method(&self) {}
                    pub(super) fn the_method(&self) {}
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
                source_range: 147..147,
                delete: 147..147,
                insert: "the_method()$0",
                kind: Method,
                lookup: "the_method",
                detail: "pub(super) fn the_method(&self)",
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
                source_range: 90..90,
                delete: 90..90,
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
                source_range: 94..94,
                delete: 94..94,
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
    fn completes_trait_method_from_other_module() {
        assert_debug_snapshot!(
            do_ref_completion(
                r"
            struct A {}
            mod m {
                pub trait Trait { fn the_method(&self); }
            }
            use m::Trait;
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
                source_range: 122..122,
                delete: 122..122,
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
                source_range: 128..128,
                delete: 128..128,
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
                source_range: 38..38,
                delete: 38..38,
                insert: "0",
                kind: Field,
                detail: "i32",
            },
            CompletionItem {
                label: "1",
                source_range: 38..38,
                delete: 38..38,
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
                source_range: 190..191,
                delete: 190..191,
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
                source_range: 69..69,
                delete: 69..69,
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
                #[lang = "future_trait"]
                pub trait Future {}
            }
            "###, CompletionKind::Keyword),
        @r###"
        [
            CompletionItem {
                label: "await",
                source_range: 74..74,
                delete: 74..74,
                insert: "await",
                detail: "expr.await",
            },
        ]
        "###
        )
    }

    #[test]
    fn test_super_super_completion() {
        assert_debug_snapshot!(
        do_ref_completion(
                r"
                mod a {
                    const A: usize = 0;

                    mod b {
                        const B: usize = 0;

                        mod c {
                            use super::super::<|>
                        }
                    }
                }
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "A",
                source_range: 120..120,
                delete: 120..120,
                insert: "A",
                kind: Const,
            },
            CompletionItem {
                label: "b",
                source_range: 120..120,
                delete: 120..120,
                insert: "b",
                kind: Module,
            },
        ]
        "###
        );
    }

    #[test]
    fn works_in_simple_macro_1() {
        assert_debug_snapshot!(
            do_ref_completion(
                r"
                macro_rules! m { ($e:expr) => { $e } }
                struct A { the_field: u32 }
                fn foo(a: A) {
                    m!(a.x<|>)
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "the_field",
                source_range: 91..92,
                delete: 91..92,
                insert: "the_field",
                kind: Field,
                detail: "u32",
            },
        ]
        "###
        );
    }

    #[test]
    fn works_in_simple_macro_recursive() {
        assert_debug_snapshot!(
            do_ref_completion(
                r"
                macro_rules! m { ($e:expr) => { $e } }
                struct A { the_field: u32 }
                fn foo(a: A) {
                    m!(a.x<|>)
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "the_field",
                source_range: 91..92,
                delete: 91..92,
                insert: "the_field",
                kind: Field,
                detail: "u32",
            },
        ]
        "###
        );
    }

    #[test]
    fn works_in_simple_macro_2() {
        // this doesn't work yet because the macro doesn't expand without the token -- maybe it can be fixed with better recovery
        assert_debug_snapshot!(
            do_ref_completion(
                r"
                macro_rules! m { ($e:expr) => { $e } }
                struct A { the_field: u32 }
                fn foo(a: A) {
                    m!(a.<|>)
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "the_field",
                source_range: 91..91,
                delete: 91..91,
                insert: "the_field",
                kind: Field,
                detail: "u32",
            },
        ]
        "###
        );
    }

    #[test]
    fn works_in_simple_macro_recursive_1() {
        assert_debug_snapshot!(
            do_ref_completion(
                r"
                macro_rules! m { ($e:expr) => { $e } }
                struct A { the_field: u32 }
                fn foo(a: A) {
                    m!(m!(m!(a.x<|>)))
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "the_field",
                source_range: 97..98,
                delete: 97..98,
                insert: "the_field",
                kind: Field,
                detail: "u32",
            },
        ]
        "###
        );
    }

    #[test]
    fn macro_expansion_resilient() {
        assert_debug_snapshot!(
            do_ref_completion(
                r"
                macro_rules! dbg {
                    () => {};
                    ($val:expr) => {
                        match $val { tmp => { tmp } }
                    };
                    // Trailing comma with single argument is ignored
                    ($val:expr,) => { $crate::dbg!($val) };
                    ($($val:expr),+ $(,)?) => {
                        ($($crate::dbg!($val)),+,)
                    };
                }
                struct A { the_field: u32 }
                fn foo(a: A) {
                    dbg!(a.<|>)
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "the_field",
                source_range: 327..327,
                delete: 327..327,
                insert: "the_field",
                kind: Field,
                detail: "u32",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_method_completion_3547() {
        assert_debug_snapshot!(
            do_ref_completion(
                r"
            struct HashSet<T> {}
            impl<T> HashSet<T> {
                pub fn the_method(&self) {}
            }
            fn foo() {
                let s: HashSet<_>;
                s.<|>
            }
            ",
            ),
            @r###"
        [
            CompletionItem {
                label: "the_method()",
                source_range: 116..116,
                delete: 116..116,
                insert: "the_method()$0",
                kind: Method,
                lookup: "the_method",
                detail: "pub fn the_method(&self)",
            },
        ]
        "###
        );
    }
}
