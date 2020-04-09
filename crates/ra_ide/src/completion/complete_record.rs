//! Complete fields in record literals and patterns.
use crate::completion::{CompletionContext, Completions};

pub(super) fn complete_record(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    let missing_fields = match (ctx.record_lit_pat.as_ref(), ctx.record_lit_syntax.as_ref()) {
        (None, None) => return None,
        (Some(_), Some(_)) => unreachable!("A record cannot be both a literal and a pattern"),
        (Some(record_pat), _) => ctx.sema.record_pattern_missing_fields(record_pat),
        (_, Some(record_lit)) => ctx.sema.record_literal_missing_fields(record_lit),
    };

    for (field, ty) in missing_fields {
        acc.add_field(ctx, field, &ty)
    }

    Some(())
}

#[cfg(test)]
mod tests {
    mod record_pat_tests {
        use insta::assert_debug_snapshot;

        use crate::completion::{test_utils::do_completion, CompletionItem, CompletionKind};

        fn complete(code: &str) -> Vec<CompletionItem> {
            do_completion(code, CompletionKind::Reference)
        }

        #[test]
        fn test_record_pattern_field() {
            let completions = complete(
                r"
            struct S { foo: u32 }

            fn process(f: S) {
                match f {
                    S { f<|>: 92 } => (),
                }
            }
            ",
            );
            assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "foo",
                source_range: [117; 118),
                delete: [117; 118),
                insert: "foo",
                kind: Field,
                detail: "u32",
            },
        ]
        "###);
        }

        #[test]
        fn test_record_pattern_enum_variant() {
            let completions = complete(
                r"
            enum E {
                S { foo: u32, bar: () }
            }

            fn process(e: E) {
                match e {
                    E::S { <|> } => (),
                }
            }
            ",
            );
            assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "bar",
                source_range: [161; 161),
                delete: [161; 161),
                insert: "bar",
                kind: Field,
                detail: "()",
            },
            CompletionItem {
                label: "foo",
                source_range: [161; 161),
                delete: [161; 161),
                insert: "foo",
                kind: Field,
                detail: "u32",
            },
        ]
        "###);
        }

        #[test]
        fn test_record_pattern_field_in_simple_macro() {
            let completions = complete(
                r"
            macro_rules! m { ($e:expr) => { $e } }
            struct S { foo: u32 }

            fn process(f: S) {
                m!(match f {
                    S { f<|>: 92 } => (),
                })
            }
            ",
            );
            assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "foo",
                source_range: [171; 172),
                delete: [171; 172),
                insert: "foo",
                kind: Field,
                detail: "u32",
            },
        ]
        "###);
        }

        #[test]
        fn only_missing_fields_are_completed_in_destruct_pats() {
            let completions = complete(
                r"
            struct S {
                foo1: u32,
                foo2: u32,
                bar: u32,
                baz: u32,
            }

            fn main() {
                let s = S {
                    foo1: 1,
                    foo2: 2,
                    bar: 3,
                    baz: 4,
                };
                if let S { foo1, foo2: a, <|> } = s {}
            }
            ",
            );
            assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "bar",
                source_range: [372; 372),
                delete: [372; 372),
                insert: "bar",
                kind: Field,
                detail: "u32",
            },
            CompletionItem {
                label: "baz",
                source_range: [372; 372),
                delete: [372; 372),
                insert: "baz",
                kind: Field,
                detail: "u32",
            },
        ]
        "###);
        }
    }

    mod record_lit_tests {
        use insta::assert_debug_snapshot;

        use crate::completion::{test_utils::do_completion, CompletionItem, CompletionKind};

        fn complete(code: &str) -> Vec<CompletionItem> {
            do_completion(code, CompletionKind::Reference)
        }

        #[test]
        fn test_record_literal_deprecated_field() {
            let completions = complete(
                r"
            struct A {
                #[deprecated]
                the_field: u32,
            }
            fn foo() {
               A { the<|> }
            }
            ",
            );
            assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "the_field",
                source_range: [142; 145),
                delete: [142; 145),
                insert: "the_field",
                kind: Field,
                detail: "u32",
                deprecated: true,
            },
        ]
        "###);
        }

        #[test]
        fn test_record_literal_field() {
            let completions = complete(
                r"
            struct A { the_field: u32 }
            fn foo() {
               A { the<|> }
            }
            ",
            );
            assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "the_field",
                source_range: [83; 86),
                delete: [83; 86),
                insert: "the_field",
                kind: Field,
                detail: "u32",
            },
        ]
        "###);
        }

        #[test]
        fn test_record_literal_enum_variant() {
            let completions = complete(
                r"
            enum E {
                A { a: u32 }
            }
            fn foo() {
                let _ = E::A { <|> }
            }
            ",
            );
            assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "a",
                source_range: [119; 119),
                delete: [119; 119),
                insert: "a",
                kind: Field,
                detail: "u32",
            },
        ]
        "###);
        }

        #[test]
        fn test_record_literal_two_structs() {
            let completions = complete(
                r"
            struct A { a: u32 }
            struct B { b: u32 }

            fn foo() {
               let _: A = B { <|> }
            }
            ",
            );
            assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "b",
                source_range: [119; 119),
                delete: [119; 119),
                insert: "b",
                kind: Field,
                detail: "u32",
            },
        ]
        "###);
        }

        #[test]
        fn test_record_literal_generic_struct() {
            let completions = complete(
                r"
            struct A<T> { a: T }

            fn foo() {
               let _: A<u32> = A { <|> }
            }
            ",
            );
            assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "a",
                source_range: [93; 93),
                delete: [93; 93),
                insert: "a",
                kind: Field,
                detail: "u32",
            },
        ]
        "###);
        }

        #[test]
        fn test_record_literal_field_in_simple_macro() {
            let completions = complete(
                r"
            macro_rules! m { ($e:expr) => { $e } }
            struct A { the_field: u32 }
            fn foo() {
               m!(A { the<|> })
            }
            ",
            );
            assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "the_field",
                source_range: [137; 140),
                delete: [137; 140),
                insert: "the_field",
                kind: Field,
                detail: "u32",
            },
        ]
        "###);
        }

        #[test]
        fn only_missing_fields_are_completed() {
            let completions = complete(
                r"
            struct S {
                foo1: u32,
                foo2: u32,
                bar: u32,
                baz: u32,
            }

            fn main() {
                let foo1 = 1;
                let s = S {
                    foo1,
                    foo2: 5,
                    <|>
                }
            }
            ",
            );
            assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "bar",
                source_range: [302; 302),
                delete: [302; 302),
                insert: "bar",
                kind: Field,
                detail: "u32",
            },
            CompletionItem {
                label: "baz",
                source_range: [302; 302),
                delete: [302; 302),
                insert: "baz",
                kind: Field,
                detail: "u32",
            },
        ]
        "###);
        }

        #[test]
        fn completes_functional_update() {
            let completions = complete(
                r"
            struct S {
                foo1: u32,
                foo2: u32,
            }

            fn main() {
                let foo1 = 1;
                let s = S {
                    foo1,
                    <|>
                    .. loop {}
                }
            }
            ",
            );
            assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "foo2",
                source_range: [221; 221),
                delete: [221; 221),
                insert: "foo2",
                kind: Field,
                detail: "u32",
            },
        ]
        "###);
        }
    }
}
