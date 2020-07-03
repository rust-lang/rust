//! Completion for attributes
//!
//! This module uses a bit of static metadata to provide completions
//! for built-in attributes.

use ra_syntax::{ast, AstNode, SyntaxKind};
use rustc_hash::FxHashSet;

use crate::completion::{
    completion_context::CompletionContext,
    completion_item::{CompletionItem, CompletionItemKind, CompletionKind, Completions},
};

pub(super) fn complete_attribute(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    let attribute = ctx.attribute_under_caret.as_ref()?;

    match (attribute.path(), attribute.input()) {
        (Some(path), Some(ast::AttrInput::TokenTree(token_tree)))
            if path.to_string() == "derive" =>
        {
            complete_derive(acc, ctx, token_tree)
        }
        (_, Some(ast::AttrInput::TokenTree(_token_tree))) => {}
        _ => complete_attribute_start(acc, ctx, attribute),
    }
    Some(())
}

fn complete_attribute_start(acc: &mut Completions, ctx: &CompletionContext, attribute: &ast::Attr) {
    for attr_completion in ATTRIBUTES {
        let mut item = CompletionItem::new(
            CompletionKind::Attribute,
            ctx.source_range(),
            attr_completion.label,
        )
        .kind(CompletionItemKind::Attribute);

        if let Some(lookup) = attr_completion.lookup {
            item = item.lookup_by(lookup);
        }

        match (attr_completion.snippet, ctx.config.snippet_cap) {
            (Some(snippet), Some(cap)) => {
                item = item.insert_snippet(cap, snippet);
            }
            _ => {}
        }

        if attribute.kind() == ast::AttrKind::Inner || !attr_completion.prefer_inner {
            acc.add(item);
        }
    }
}

struct AttrCompletion {
    label: &'static str,
    lookup: Option<&'static str>,
    snippet: Option<&'static str>,
    prefer_inner: bool,
}

impl AttrCompletion {
    const fn prefer_inner(self) -> AttrCompletion {
        AttrCompletion { prefer_inner: true, ..self }
    }
}

const fn attr(
    label: &'static str,
    lookup: Option<&'static str>,
    snippet: Option<&'static str>,
) -> AttrCompletion {
    AttrCompletion { label, lookup, snippet, prefer_inner: false }
}

const ATTRIBUTES: &[AttrCompletion] = &[
    attr("allow(…)", Some("allow"), Some("allow(${0:lint})")),
    attr("cfg_attr(…)", Some("cfg_attr"), Some("cfg_attr(${1:predicate}, ${0:attr})")),
    attr("cfg(…)", Some("cfg"), Some("cfg(${0:predicate})")),
    attr("deny(…)", Some("deny"), Some("deny(${0:lint})")),
    attr(r#"deprecated = "…""#, Some("deprecated"), Some(r#"deprecated = "${0:reason}""#)),
    attr("derive(…)", Some("derive"), Some(r#"derive(${0:Debug})"#)),
    attr(r#"doc = "…""#, Some("doc"), Some(r#"doc = "${0:docs}""#)),
    attr("feature(…)", Some("feature"), Some("feature(${0:flag})")).prefer_inner(),
    attr("forbid(…)", Some("forbid"), Some("forbid(${0:lint})")),
    // FIXME: resolve through macro resolution?
    attr("global_allocator", None, None).prefer_inner(),
    attr("ignore(…)", Some("ignore"), Some("ignore(${0:lint})")),
    attr("inline(…)", Some("inline"), Some("inline(${0:lint})")),
    attr(r#"link_name = "…""#, Some("link_name"), Some(r#"link_name = "${0:symbol_name}""#)),
    attr("link", None, None),
    attr("macro_export", None, None),
    attr("macro_use", None, None),
    attr(r#"must_use = "…""#, Some("must_use"), Some(r#"must_use = "${0:reason}""#)),
    attr("no_mangle", None, None),
    attr("no_std", None, None).prefer_inner(),
    attr("non_exhaustive", None, None),
    attr("panic_handler", None, None).prefer_inner(),
    attr("path = \"…\"", Some("path"), Some("path =\"${0:path}\"")),
    attr("proc_macro", None, None),
    attr("proc_macro_attribute", None, None),
    attr("proc_macro_derive(…)", Some("proc_macro_derive"), Some("proc_macro_derive(${0:Trait})")),
    attr("recursion_limit = …", Some("recursion_limit"), Some("recursion_limit = ${0:128}"))
        .prefer_inner(),
    attr("repr(…)", Some("repr"), Some("repr(${0:C})")),
    attr(
        "should_panic(…)",
        Some("should_panic"),
        Some(r#"should_panic(expected = "${0:reason}")"#),
    ),
    attr(
        r#"target_feature = "…""#,
        Some("target_feature"),
        Some("target_feature = \"${0:feature}\""),
    ),
    attr("test", None, None),
    attr("used", None, None),
    attr("warn(…)", Some("warn"), Some("warn(${0:lint})")),
    attr(
        r#"windows_subsystem = "…""#,
        Some("windows_subsystem"),
        Some(r#"windows_subsystem = "${0:subsystem}""#),
    )
    .prefer_inner(),
];

fn complete_derive(acc: &mut Completions, ctx: &CompletionContext, derive_input: ast::TokenTree) {
    if let Ok(existing_derives) = parse_derive_input(derive_input) {
        for derive_completion in DEFAULT_DERIVE_COMPLETIONS
            .into_iter()
            .filter(|completion| !existing_derives.contains(completion.label))
        {
            let mut label = derive_completion.label.to_owned();
            for dependency in derive_completion
                .dependencies
                .into_iter()
                .filter(|&&dependency| !existing_derives.contains(dependency))
            {
                label.push_str(", ");
                label.push_str(dependency);
            }
            acc.add(
                CompletionItem::new(CompletionKind::Attribute, ctx.source_range(), label)
                    .kind(CompletionItemKind::Attribute),
            );
        }

        for custom_derive_name in get_derive_names_in_scope(ctx).difference(&existing_derives) {
            acc.add(
                CompletionItem::new(
                    CompletionKind::Attribute,
                    ctx.source_range(),
                    custom_derive_name,
                )
                .kind(CompletionItemKind::Attribute),
            );
        }
    }
}

fn parse_derive_input(derive_input: ast::TokenTree) -> Result<FxHashSet<String>, ()> {
    match (derive_input.left_delimiter_token(), derive_input.right_delimiter_token()) {
        (Some(left_paren), Some(right_paren))
            if left_paren.kind() == SyntaxKind::L_PAREN
                && right_paren.kind() == SyntaxKind::R_PAREN =>
        {
            let mut input_derives = FxHashSet::default();
            let mut current_derive = String::new();
            for token in derive_input
                .syntax()
                .children_with_tokens()
                .filter_map(|token| token.into_token())
                .skip_while(|token| token != &left_paren)
                .skip(1)
                .take_while(|token| token != &right_paren)
            {
                if SyntaxKind::COMMA == token.kind() {
                    if !current_derive.is_empty() {
                        input_derives.insert(current_derive);
                        current_derive = String::new();
                    }
                } else {
                    current_derive.push_str(token.to_string().trim());
                }
            }

            if !current_derive.is_empty() {
                input_derives.insert(current_derive);
            }
            Ok(input_derives)
        }
        _ => Err(()),
    }
}

fn get_derive_names_in_scope(ctx: &CompletionContext) -> FxHashSet<String> {
    let mut result = FxHashSet::default();
    ctx.scope().process_all_names(&mut |name, scope_def| {
        if let hir::ScopeDef::MacroDef(mac) = scope_def {
            if mac.is_derive_macro() {
                result.insert(name.to_string());
            }
        }
    });
    result
}

struct DeriveCompletion {
    label: &'static str,
    dependencies: &'static [&'static str],
}

/// Standard Rust derives and the information about their dependencies
/// (the dependencies are needed so that the main derive don't break the compilation when added)
const DEFAULT_DERIVE_COMPLETIONS: &[DeriveCompletion] = &[
    DeriveCompletion { label: "Clone", dependencies: &[] },
    DeriveCompletion { label: "Copy", dependencies: &["Clone"] },
    DeriveCompletion { label: "Debug", dependencies: &[] },
    DeriveCompletion { label: "Default", dependencies: &[] },
    DeriveCompletion { label: "Hash", dependencies: &[] },
    DeriveCompletion { label: "PartialEq", dependencies: &[] },
    DeriveCompletion { label: "Eq", dependencies: &["PartialEq"] },
    DeriveCompletion { label: "PartialOrd", dependencies: &["PartialEq"] },
    DeriveCompletion { label: "Ord", dependencies: &["PartialOrd", "Eq", "PartialEq"] },
];

#[cfg(test)]
mod tests {
    use expect::{expect, Expect};

    use crate::completion::{test_utils::completion_list, CompletionKind};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Attribute);
        expect.assert_eq(&actual);
    }

    #[test]
    fn empty_derive_completion() {
        check(
            r#"
#[derive(<|>)]
struct Test {}
        "#,
            expect![[r#"
                at Clone
                at Copy, Clone
                at Debug
                at Default
                at Eq, PartialEq
                at Hash
                at Ord, PartialOrd, Eq, PartialEq
                at PartialEq
                at PartialOrd, PartialEq
            "#]],
        );
    }

    #[test]
    fn no_completion_for_incorrect_derive() {
        check(
            r#"
#[derive{<|>)]
struct Test {}
"#,
            expect![[r#""#]],
        )
    }

    #[test]
    fn derive_with_input_completion() {
        check(
            r#"
#[derive(serde::Serialize, PartialEq, <|>)]
struct Test {}
"#,
            expect![[r#"
                at Clone
                at Copy, Clone
                at Debug
                at Default
                at Eq
                at Hash
                at Ord, PartialOrd, Eq
                at PartialOrd
            "#]],
        )
    }

    #[test]
    fn test_attribute_completion() {
        check(
            r#"#[<|>]"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at deprecated = "…"
                at derive(…)
                at doc = "…"
                at forbid(…)
                at ignore(…)
                at inline(…)
                at link
                at link_name = "…"
                at macro_export
                at macro_use
                at must_use = "…"
                at no_mangle
                at non_exhaustive
                at path = "…"
                at proc_macro
                at proc_macro_attribute
                at proc_macro_derive(…)
                at repr(…)
                at should_panic(…)
                at target_feature = "…"
                at test
                at used
                at warn(…)
            "#]],
        )
    }

    #[test]
    fn test_attribute_completion_inside_nested_attr() {
        check(r#"#[allow(<|>)]"#, expect![[]])
    }

    #[test]
    fn test_inner_attribute_completion() {
        check(
            r"#![<|>]",
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at deprecated = "…"
                at derive(…)
                at doc = "…"
                at feature(…)
                at forbid(…)
                at global_allocator
                at ignore(…)
                at inline(…)
                at link
                at link_name = "…"
                at macro_export
                at macro_use
                at must_use = "…"
                at no_mangle
                at no_std
                at non_exhaustive
                at panic_handler
                at path = "…"
                at proc_macro
                at proc_macro_attribute
                at proc_macro_derive(…)
                at recursion_limit = …
                at repr(…)
                at should_panic(…)
                at target_feature = "…"
                at test
                at used
                at warn(…)
                at windows_subsystem = "…"
            "#]],
        );
    }
}
