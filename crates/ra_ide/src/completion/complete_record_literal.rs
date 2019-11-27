//! FIXME: write short doc here

use crate::completion::{CompletionContext, Completions};

/// Complete fields in fields literals.
pub(super) fn complete_record_literal(acc: &mut Completions, ctx: &CompletionContext) {
    let (ty, variant) = match ctx.record_lit_syntax.as_ref().and_then(|it| {
        Some((
            ctx.analyzer.type_of(ctx.db, &it.clone().into())?,
            ctx.analyzer.resolve_record_literal(it)?,
        ))
    }) {
        Some(it) => it,
        _ => return,
    };

    for (field, field_ty) in ty.variant_fields(ctx.db, variant) {
        acc.add_field(ctx, field, &field_ty);
    }
}

#[cfg(test)]
mod tests {
    use crate::completion::{do_completion, CompletionItem, CompletionKind};
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
}
