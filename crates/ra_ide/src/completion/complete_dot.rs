//! FIXME: write short doc here

use hir::{
    HasVisibility,
    // HirDisplay,
    Type,
};

use crate::completion::completion_item::CompletionKind;
use crate::{
    call_info::call_info,
    completion::{
        completion_context::CompletionContext,
        completion_item::{Completions, SortOption},
    },
    // CallInfo,
    CompletionItem,
};
use rustc_hash::FxHashSet;
// use std::cmp::Ordering;

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
        let fields = receiver.fields(ctx.db);

        // If we use this implementation we can delete call_info in the CompletionContext
        if let Some(call_info) = call_info(ctx.db, ctx.file_position) {
            acc.with_sort_option(SortOption::CallFn(call_info));
        }

        // // For Call Fn
        // if let Some(call_info) = &ctx.call_info {
        //     if let Some(active_parameter_type) = call_info.active_parameter_type() {
        //         let active_parameter_name = call_info.active_parameter_name().unwrap();
        //         fields.sort_by(|a, b| {
        //             // For the same type
        //             if active_parameter_type == a.1.display(ctx.db).to_string() {
        //                 // If same type + same name then go top position
        //                 if active_parameter_name == a.0.name(ctx.db).to_string() {
        //                     Ordering::Less
        //                 } else {
        //                     if active_parameter_type == b.1.display(ctx.db).to_string() {
        //                         Ordering::Equal
        //                     } else {
        //                         Ordering::Less
        //                     }
        //                 }
        //             } else {
        //                 Ordering::Greater
        //             }
        //         });
        //     }
        // }

        // For Lit struct fields
        // ---

        for (field, ty) in fields {
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
    use crate::completion::{
        test_utils::{do_completion, do_completion_without_sort},
        CompletionItem, CompletionKind,
    };
    use insta::assert_debug_snapshot;

    fn do_ref_completion(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Reference)
    }

    fn do_ref_completion_without_sort(code: &str) -> Vec<CompletionItem> {
        do_completion_without_sort(code, CompletionKind::Reference)
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
    fn test_struct_field_completion_in_func_call() {
        assert_debug_snapshot!(
        do_ref_completion_without_sort(
                r"
                struct A { another_field: i64, the_field: u32, my_string: String }
                fn test(my_param: u32) -> u32 { my_param }
                fn foo(a: A) {
                    test(a.<|>)
                }
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "the_field",
                source_range: [201; 201),
                delete: [201; 201),
                insert: "the_field",
                kind: Field,
                detail: "u32",
            },
            CompletionItem {
                label: "another_field",
                source_range: [201; 201),
                delete: [201; 201),
                insert: "another_field",
                kind: Field,
                detail: "i64",
            },
            CompletionItem {
                label: "my_string",
                source_range: [201; 201),
                delete: [201; 201),
                insert: "my_string",
                kind: Field,
                detail: "{unknown}",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_struct_field_completion_in_func_call_with_type_and_name() {
        assert_debug_snapshot!(
        do_ref_completion_without_sort(
                r"
                struct A { another_field: i64, another_good_type: u32, the_field: u32 }
                fn test(the_field: u32) -> u32 { the_field }
                fn foo(a: A) {
                    test(a.<|>)
                }
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "the_field",
                source_range: [208; 208),
                delete: [208; 208),
                insert: "the_field",
                kind: Field,
                detail: "u32",
            },
            CompletionItem {
                label: "another_good_type",
                source_range: [208; 208),
                delete: [208; 208),
                insert: "another_good_type",
                kind: Field,
                detail: "u32",
            },
            CompletionItem {
                label: "another_field",
                source_range: [208; 208),
                delete: [208; 208),
                insert: "another_field",
                kind: Field,
                detail: "i64",
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
                source_range: [256; 256),
                delete: [256; 256),
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
                source_range: [219; 219),
                delete: [219; 219),
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
                #[lang = "future_trait"]
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
                source_range: [217; 217),
                delete: [217; 217),
                insert: "A",
                kind: Const,
            },
            CompletionItem {
                label: "b",
                source_range: [217; 217),
                delete: [217; 217),
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
                source_range: [156; 157),
                delete: [156; 157),
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
                source_range: [156; 157),
                delete: [156; 157),
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
                source_range: [156; 156),
                delete: [156; 156),
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
                source_range: [162; 163),
                delete: [162; 163),
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
                source_range: [552; 552),
                delete: [552; 552),
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
                source_range: [201; 201),
                delete: [201; 201),
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
