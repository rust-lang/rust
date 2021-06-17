//! Completion for representations.

use syntax::ast;

use crate::{
    context::CompletionContext,
    item::{CompletionItem, CompletionItemKind, CompletionKind},
    Completions,
};

pub(super) fn complete_repr(
    acc: &mut Completions,
    ctx: &CompletionContext,
    derive_input: ast::TokenTree,
) {
    if let Some(existing_reprs) = super::parse_comma_sep_input(derive_input) {
        for repr_completion in REPR_COMPLETIONS {
            if existing_reprs
                .iter()
                .any(|it| repr_completion.label == it || repr_completion.collides.contains(&&**it))
            {
                continue;
            }
            let mut item = CompletionItem::new(
                CompletionKind::Attribute,
                ctx.source_range(),
                repr_completion.label,
            );
            item.kind(CompletionItemKind::Attribute);
            if let Some(lookup) = repr_completion.lookup {
                item.lookup_by(lookup);
            }
            if let Some((snippet, cap)) = repr_completion.snippet.zip(ctx.config.snippet_cap) {
                item.insert_snippet(cap, snippet);
            }
            item.add_to(acc);
        }
    }
}

struct ReprCompletion {
    label: &'static str,
    snippet: Option<&'static str>,
    lookup: Option<&'static str>,
    collides: &'static [&'static str],
}

const fn attr(label: &'static str, collides: &'static [&'static str]) -> ReprCompletion {
    ReprCompletion { label, snippet: None, lookup: None, collides }
}

#[rustfmt::skip]
const REPR_COMPLETIONS: &[ReprCompletion] = &[
    ReprCompletion { label: "align($0)", snippet: Some("align($0)"), lookup: Some("align"), collides: &["transparent", "packed"] },
    attr("packed", &["transparent", "align"]),
    attr("transparent", &["C", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize"]),
    attr("C", &["transparent"]),
    attr("u8",     &["transparent", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize"]),
    attr("u16",    &["transparent", "u8", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize"]),
    attr("u32",    &["transparent", "u8", "u16", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize"]),
    attr("u64",    &["transparent", "u8", "u16", "u32", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize"]),
    attr("u128",   &["transparent", "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "i128", "isize"]),
    attr("usize",  &["transparent", "u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32", "i64", "i128", "isize"]),
    attr("i8",     &["transparent", "u8", "u16", "u32", "u64", "u128", "usize", "i16", "i32", "i64", "i128", "isize"]),
    attr("i16",    &["transparent", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i32", "i64", "i128", "isize"]),
    attr("i32",    &["transparent", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i64", "i128", "isize"]),
    attr("i64",    &["transparent", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i128", "isize"]),
    attr("i28",    &["transparent", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "isize"]),
    attr("isize",  &["transparent", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128"]),
];

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::tests::completion_list;

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture);
        expect.assert_eq(&actual);
    }

    #[test]
    fn no_completion_for_incorrect_repr() {
        check(r#"#[repr{$0)] struct Test;"#, expect![[]])
    }

    #[test]
    fn empty() {
        check(
            r#"#[repr($0)] struct Test;"#,
            expect![[r#"
                at align($0)
                at packed
                at transparent
                at C
                at u8
                at u16
                at u32
                at u64
                at u128
                at usize
                at i8
                at i16
                at i32
                at i64
                at i28
                at isize
            "#]],
        );
    }

    #[test]
    fn transparent() {
        check(r#"#[repr(transparent, $0)] struct Test;"#, expect![[r#""#]]);
    }

    #[test]
    fn align() {
        check(
            r#"#[repr(align(1), $0)] struct Test;"#,
            expect![[r#"
                at align($0)
                at transparent
                at C
                at u8
                at u16
                at u32
                at u64
                at u128
                at usize
                at i8
                at i16
                at i32
                at i64
                at i28
                at isize
            "#]],
        );
    }

    #[test]
    fn packed() {
        check(
            r#"#[repr(packed, $0)] struct Test;"#,
            expect![[r#"
                at transparent
                at C
                at u8
                at u16
                at u32
                at u64
                at u128
                at usize
                at i8
                at i16
                at i32
                at i64
                at i28
                at isize
            "#]],
        );
    }

    #[test]
    fn c() {
        check(
            r#"#[repr(C, $0)] struct Test;"#,
            expect![[r#"
                at align($0)
                at packed
                at u8
                at u16
                at u32
                at u64
                at u128
                at usize
                at i8
                at i16
                at i32
                at i64
                at i28
                at isize
            "#]],
        );
    }

    #[test]
    fn prim() {
        check(
            r#"#[repr(usize, $0)] struct Test;"#,
            expect![[r#"
                at align($0)
                at packed
                at C
            "#]],
        );
    }
}
