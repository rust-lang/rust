//! FIXME: write short doc here

use crate::completion::{CompletionContext, Completions};
use ra_syntax::{ast::NameOwner, SmolStr};

pub(super) fn complete_record_pattern(acc: &mut Completions, ctx: &CompletionContext) {
    let (ty, variant) = match ctx.record_lit_pat.as_ref().and_then(|it| {
        Some((ctx.sema.type_of_pat(&it.clone().into())?, ctx.sema.resolve_record_pattern(it)?))
    }) {
        Some(it) => it,
        _ => return,
    };

    let already_present_names: Vec<SmolStr> = ctx
        .record_lit_pat
        .as_ref()
        .and_then(|record_pat| record_pat.record_field_pat_list())
        .map(|pat_list| pat_list.bind_pats())
        .map(|bind_pats| {
            bind_pats
                .into_iter()
                .filter_map(|pat| pat.name())
                .map(|name| name.text().clone())
                .collect()
        })
        .unwrap_or_default();

    for (field, field_ty) in ty.variant_fields(ctx.db, variant) {
        if !already_present_names.contains(&SmolStr::from(field.name(ctx.db).to_string())) {
            acc.add_field(ctx, field, &field_ty);
        }
    }
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
                if let S { foo1, foo2, <|> } = s {}
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "bar",
                source_range: [369; 369),
                delete: [369; 369),
                insert: "bar",
                kind: Field,
                detail: "u32",
            },
            CompletionItem {
                label: "baz",
                source_range: [369; 369),
                delete: [369; 369),
                insert: "baz",
                kind: Field,
                detail: "u32",
            },
        ]
        "###);
    }
}
