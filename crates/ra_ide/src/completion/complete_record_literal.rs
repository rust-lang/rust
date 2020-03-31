//! FIXME: write short doc here

use super::get_missing_fields;
use crate::completion::{CompletionContext, Completions};
use either::Either;

/// Complete fields in fields literals.
pub(super) fn complete_record_literal(
    acc: &mut Completions,
    ctx: &CompletionContext,
) -> Option<()> {
    let record_lit = ctx.record_lit_syntax.as_ref()?;
    for (field, field_ty) in get_missing_fields(ctx, Either::Left(record_lit))? {
        acc.add_field(ctx, field, &field_ty);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use crate::completion::{test_utils::do_completion, CompletionItem, CompletionKind};
    use insta::assert_debug_snapshot;

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
}
