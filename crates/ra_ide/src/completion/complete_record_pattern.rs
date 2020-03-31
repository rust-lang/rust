//! FIXME: write short doc here

use super::get_missing_fields;
use crate::completion::{CompletionContext, Completions};
use either::Either;

pub(super) fn complete_record_pattern(
    acc: &mut Completions,
    ctx: &CompletionContext,
) -> Option<()> {
    let record_pat = ctx.record_lit_pat.as_ref()?;
    for (field, field_ty) in get_missing_fields(ctx, Either::Right(record_pat))? {
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
