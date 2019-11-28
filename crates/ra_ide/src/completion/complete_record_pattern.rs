//! FIXME: write short doc here

use crate::completion::{CompletionContext, Completions};

pub(super) fn complete_record_pattern(acc: &mut Completions, ctx: &CompletionContext) {
    let (ty, variant) = match ctx.record_lit_pat.as_ref().and_then(|it| {
        Some((
            ctx.analyzer.type_of_pat(ctx.db, &it.clone().into())?,
            ctx.analyzer.resolve_record_pattern(it)?,
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
}
