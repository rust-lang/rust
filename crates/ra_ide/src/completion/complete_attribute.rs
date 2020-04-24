//! Completion for attributes
//!
//! This module uses a bit of static metadata to provide completions
//! for built-in attributes.

use super::completion_context::CompletionContext;
use super::completion_item::{CompletionItem, CompletionItemKind, CompletionKind, Completions};
use ra_syntax::{
    ast::{Attr, AttrKind},
    AstNode,
};

pub(super) fn complete_attribute(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.is_attribute {
        return;
    }

    let is_inner = ctx
        .original_token
        .ancestors()
        .find_map(Attr::cast)
        .map(|attr| attr.kind() == AttrKind::Inner)
        .unwrap_or(false);

    for attr_completion in ATTRIBUTES {
        let mut item = CompletionItem::new(
            CompletionKind::Attribute,
            ctx.source_range(),
            attr_completion.label,
        )
        .kind(CompletionItemKind::Attribute);

        match (attr_completion.snippet, ctx.config.snippet_cap) {
            (Some(snippet), Some(cap)) => {
                item = item.insert_snippet(cap, snippet);
            }
            _ => {}
        }

        if is_inner || !attr_completion.should_be_inner {
            acc.add(item);
        }
    }
}

struct AttrCompletion {
    label: &'static str,
    snippet: Option<&'static str>,
    should_be_inner: bool,
}

const ATTRIBUTES: &[AttrCompletion] = &[
    AttrCompletion { label: "allow", snippet: Some("allow(${0:lint})"), should_be_inner: false },
    AttrCompletion {
        label: "cfg_attr",
        snippet: Some("cfg_attr(${1:predicate}, ${0:attr})"),
        should_be_inner: false,
    },
    AttrCompletion { label: "cfg", snippet: Some("cfg(${0:predicate})"), should_be_inner: false },
    AttrCompletion { label: "deny", snippet: Some("deny(${0:lint})"), should_be_inner: false },
    AttrCompletion {
        label: "deprecated",
        snippet: Some(r#"deprecated = "${0:reason}""#),
        should_be_inner: false,
    },
    AttrCompletion {
        label: "derive",
        snippet: Some(r#"derive(${0:Debug})"#),
        should_be_inner: false,
    },
    AttrCompletion { label: "doc", snippet: Some(r#"doc = "${0:docs}""#), should_be_inner: false },
    AttrCompletion { label: "feature", snippet: Some("feature(${0:flag})"), should_be_inner: true },
    AttrCompletion { label: "forbid", snippet: Some("forbid(${0:lint})"), should_be_inner: false },
    // FIXME: resolve through macro resolution?
    AttrCompletion { label: "global_allocator", snippet: None, should_be_inner: true },
    AttrCompletion { label: "ignore", snippet: Some("ignore(${0:lint})"), should_be_inner: false },
    AttrCompletion { label: "inline", snippet: Some("inline(${0:lint})"), should_be_inner: false },
    AttrCompletion {
        label: "link_name",
        snippet: Some(r#"link_name = "${0:symbol_name}""#),
        should_be_inner: false,
    },
    AttrCompletion { label: "link", snippet: None, should_be_inner: false },
    AttrCompletion { label: "macro_export", snippet: None, should_be_inner: false },
    AttrCompletion { label: "macro_use", snippet: None, should_be_inner: false },
    AttrCompletion {
        label: "must_use",
        snippet: Some(r#"must_use = "${0:reason}""#),
        should_be_inner: false,
    },
    AttrCompletion { label: "no_mangle", snippet: None, should_be_inner: false },
    AttrCompletion { label: "no_std", snippet: None, should_be_inner: true },
    AttrCompletion { label: "non_exhaustive", snippet: None, should_be_inner: false },
    AttrCompletion { label: "panic_handler", snippet: None, should_be_inner: true },
    AttrCompletion { label: "path", snippet: Some("path =\"${0:path}\""), should_be_inner: false },
    AttrCompletion { label: "proc_macro", snippet: None, should_be_inner: false },
    AttrCompletion { label: "proc_macro_attribute", snippet: None, should_be_inner: false },
    AttrCompletion {
        label: "proc_macro_derive",
        snippet: Some("proc_macro_derive(${0:Trait})"),
        should_be_inner: false,
    },
    AttrCompletion {
        label: "recursion_limit",
        snippet: Some("recursion_limit = ${0:128}"),
        should_be_inner: true,
    },
    AttrCompletion { label: "repr", snippet: Some("repr(${0:C})"), should_be_inner: false },
    AttrCompletion {
        label: "should_panic",
        snippet: Some(r#"expected = "${0:reason}""#),
        should_be_inner: false,
    },
    AttrCompletion {
        label: "target_feature",
        snippet: Some("target_feature = \"${0:feature}\""),
        should_be_inner: false,
    },
    AttrCompletion { label: "test", snippet: None, should_be_inner: false },
    AttrCompletion { label: "used", snippet: None, should_be_inner: false },
    AttrCompletion { label: "warn", snippet: Some("warn(${0:lint})"), should_be_inner: false },
    AttrCompletion {
        label: "windows_subsystem",
        snippet: Some(r#"windows_subsystem = "${0:subsystem}""#),
        should_be_inner: true,
    },
];

#[cfg(test)]
mod tests {
    use crate::completion::{test_utils::do_completion, CompletionItem, CompletionKind};
    use insta::assert_debug_snapshot;

    fn do_attr_completion(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Attribute)
    }

    #[test]
    fn test_attribute_completion() {
        assert_debug_snapshot!(
        do_attr_completion(
                r"
                #[<|>]
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "allow",
                source_range: 19..19,
                delete: 19..19,
                insert: "allow(${0:lint})",
                kind: Attribute,
            },
            CompletionItem {
                label: "cfg",
                source_range: 19..19,
                delete: 19..19,
                insert: "cfg(${0:predicate})",
                kind: Attribute,
            },
            CompletionItem {
                label: "cfg_attr",
                source_range: 19..19,
                delete: 19..19,
                insert: "cfg_attr(${1:predicate}, ${0:attr})",
                kind: Attribute,
            },
            CompletionItem {
                label: "deny",
                source_range: 19..19,
                delete: 19..19,
                insert: "deny(${0:lint})",
                kind: Attribute,
            },
            CompletionItem {
                label: "deprecated",
                source_range: 19..19,
                delete: 19..19,
                insert: "deprecated = \"${0:reason}\"",
                kind: Attribute,
            },
            CompletionItem {
                label: "derive",
                source_range: 19..19,
                delete: 19..19,
                insert: "derive(${0:Debug})",
                kind: Attribute,
            },
            CompletionItem {
                label: "doc",
                source_range: 19..19,
                delete: 19..19,
                insert: "doc = \"${0:docs}\"",
                kind: Attribute,
            },
            CompletionItem {
                label: "forbid",
                source_range: 19..19,
                delete: 19..19,
                insert: "forbid(${0:lint})",
                kind: Attribute,
            },
            CompletionItem {
                label: "ignore",
                source_range: 19..19,
                delete: 19..19,
                insert: "ignore(${0:lint})",
                kind: Attribute,
            },
            CompletionItem {
                label: "inline",
                source_range: 19..19,
                delete: 19..19,
                insert: "inline(${0:lint})",
                kind: Attribute,
            },
            CompletionItem {
                label: "link",
                source_range: 19..19,
                delete: 19..19,
                insert: "link",
                kind: Attribute,
            },
            CompletionItem {
                label: "link_name",
                source_range: 19..19,
                delete: 19..19,
                insert: "link_name = \"${0:symbol_name}\"",
                kind: Attribute,
            },
            CompletionItem {
                label: "macro_export",
                source_range: 19..19,
                delete: 19..19,
                insert: "macro_export",
                kind: Attribute,
            },
            CompletionItem {
                label: "macro_use",
                source_range: 19..19,
                delete: 19..19,
                insert: "macro_use",
                kind: Attribute,
            },
            CompletionItem {
                label: "must_use",
                source_range: 19..19,
                delete: 19..19,
                insert: "must_use = \"${0:reason}\"",
                kind: Attribute,
            },
            CompletionItem {
                label: "no_mangle",
                source_range: 19..19,
                delete: 19..19,
                insert: "no_mangle",
                kind: Attribute,
            },
            CompletionItem {
                label: "non_exhaustive",
                source_range: 19..19,
                delete: 19..19,
                insert: "non_exhaustive",
                kind: Attribute,
            },
            CompletionItem {
                label: "path",
                source_range: 19..19,
                delete: 19..19,
                insert: "path =\"${0:path}\"",
                kind: Attribute,
            },
            CompletionItem {
                label: "proc_macro",
                source_range: 19..19,
                delete: 19..19,
                insert: "proc_macro",
                kind: Attribute,
            },
            CompletionItem {
                label: "proc_macro_attribute",
                source_range: 19..19,
                delete: 19..19,
                insert: "proc_macro_attribute",
                kind: Attribute,
            },
            CompletionItem {
                label: "proc_macro_derive",
                source_range: 19..19,
                delete: 19..19,
                insert: "proc_macro_derive(${0:Trait})",
                kind: Attribute,
            },
            CompletionItem {
                label: "repr",
                source_range: 19..19,
                delete: 19..19,
                insert: "repr(${0:C})",
                kind: Attribute,
            },
            CompletionItem {
                label: "should_panic",
                source_range: 19..19,
                delete: 19..19,
                insert: "expected = \"${0:reason}\"",
                kind: Attribute,
            },
            CompletionItem {
                label: "target_feature",
                source_range: 19..19,
                delete: 19..19,
                insert: "target_feature = \"${0:feature}\"",
                kind: Attribute,
            },
            CompletionItem {
                label: "test",
                source_range: 19..19,
                delete: 19..19,
                insert: "test",
                kind: Attribute,
            },
            CompletionItem {
                label: "used",
                source_range: 19..19,
                delete: 19..19,
                insert: "used",
                kind: Attribute,
            },
            CompletionItem {
                label: "warn",
                source_range: 19..19,
                delete: 19..19,
                insert: "warn(${0:lint})",
                kind: Attribute,
            },
        ]
        "###
        );
    }

    #[test]
    fn test_inner_attribute_completion() {
        assert_debug_snapshot!(
        do_attr_completion(
                r"
                #![<|>]
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "allow",
                source_range: 20..20,
                delete: 20..20,
                insert: "allow(${0:lint})",
                kind: Attribute,
            },
            CompletionItem {
                label: "cfg",
                source_range: 20..20,
                delete: 20..20,
                insert: "cfg(${0:predicate})",
                kind: Attribute,
            },
            CompletionItem {
                label: "cfg_attr",
                source_range: 20..20,
                delete: 20..20,
                insert: "cfg_attr(${1:predicate}, ${0:attr})",
                kind: Attribute,
            },
            CompletionItem {
                label: "deny",
                source_range: 20..20,
                delete: 20..20,
                insert: "deny(${0:lint})",
                kind: Attribute,
            },
            CompletionItem {
                label: "deprecated",
                source_range: 20..20,
                delete: 20..20,
                insert: "deprecated = \"${0:reason}\"",
                kind: Attribute,
            },
            CompletionItem {
                label: "derive",
                source_range: 20..20,
                delete: 20..20,
                insert: "derive(${0:Debug})",
                kind: Attribute,
            },
            CompletionItem {
                label: "doc",
                source_range: 20..20,
                delete: 20..20,
                insert: "doc = \"${0:docs}\"",
                kind: Attribute,
            },
            CompletionItem {
                label: "feature",
                source_range: 20..20,
                delete: 20..20,
                insert: "feature(${0:flag})",
                kind: Attribute,
            },
            CompletionItem {
                label: "forbid",
                source_range: 20..20,
                delete: 20..20,
                insert: "forbid(${0:lint})",
                kind: Attribute,
            },
            CompletionItem {
                label: "global_allocator",
                source_range: 20..20,
                delete: 20..20,
                insert: "global_allocator",
                kind: Attribute,
            },
            CompletionItem {
                label: "ignore",
                source_range: 20..20,
                delete: 20..20,
                insert: "ignore(${0:lint})",
                kind: Attribute,
            },
            CompletionItem {
                label: "inline",
                source_range: 20..20,
                delete: 20..20,
                insert: "inline(${0:lint})",
                kind: Attribute,
            },
            CompletionItem {
                label: "link",
                source_range: 20..20,
                delete: 20..20,
                insert: "link",
                kind: Attribute,
            },
            CompletionItem {
                label: "link_name",
                source_range: 20..20,
                delete: 20..20,
                insert: "link_name = \"${0:symbol_name}\"",
                kind: Attribute,
            },
            CompletionItem {
                label: "macro_export",
                source_range: 20..20,
                delete: 20..20,
                insert: "macro_export",
                kind: Attribute,
            },
            CompletionItem {
                label: "macro_use",
                source_range: 20..20,
                delete: 20..20,
                insert: "macro_use",
                kind: Attribute,
            },
            CompletionItem {
                label: "must_use",
                source_range: 20..20,
                delete: 20..20,
                insert: "must_use = \"${0:reason}\"",
                kind: Attribute,
            },
            CompletionItem {
                label: "no_mangle",
                source_range: 20..20,
                delete: 20..20,
                insert: "no_mangle",
                kind: Attribute,
            },
            CompletionItem {
                label: "no_std",
                source_range: 20..20,
                delete: 20..20,
                insert: "no_std",
                kind: Attribute,
            },
            CompletionItem {
                label: "non_exhaustive",
                source_range: 20..20,
                delete: 20..20,
                insert: "non_exhaustive",
                kind: Attribute,
            },
            CompletionItem {
                label: "panic_handler",
                source_range: 20..20,
                delete: 20..20,
                insert: "panic_handler",
                kind: Attribute,
            },
            CompletionItem {
                label: "path",
                source_range: 20..20,
                delete: 20..20,
                insert: "path =\"${0:path}\"",
                kind: Attribute,
            },
            CompletionItem {
                label: "proc_macro",
                source_range: 20..20,
                delete: 20..20,
                insert: "proc_macro",
                kind: Attribute,
            },
            CompletionItem {
                label: "proc_macro_attribute",
                source_range: 20..20,
                delete: 20..20,
                insert: "proc_macro_attribute",
                kind: Attribute,
            },
            CompletionItem {
                label: "proc_macro_derive",
                source_range: 20..20,
                delete: 20..20,
                insert: "proc_macro_derive(${0:Trait})",
                kind: Attribute,
            },
            CompletionItem {
                label: "recursion_limit",
                source_range: 20..20,
                delete: 20..20,
                insert: "recursion_limit = ${0:128}",
                kind: Attribute,
            },
            CompletionItem {
                label: "repr",
                source_range: 20..20,
                delete: 20..20,
                insert: "repr(${0:C})",
                kind: Attribute,
            },
            CompletionItem {
                label: "should_panic",
                source_range: 20..20,
                delete: 20..20,
                insert: "expected = \"${0:reason}\"",
                kind: Attribute,
            },
            CompletionItem {
                label: "target_feature",
                source_range: 20..20,
                delete: 20..20,
                insert: "target_feature = \"${0:feature}\"",
                kind: Attribute,
            },
            CompletionItem {
                label: "test",
                source_range: 20..20,
                delete: 20..20,
                insert: "test",
                kind: Attribute,
            },
            CompletionItem {
                label: "used",
                source_range: 20..20,
                delete: 20..20,
                insert: "used",
                kind: Attribute,
            },
            CompletionItem {
                label: "warn",
                source_range: 20..20,
                delete: 20..20,
                insert: "warn(${0:lint})",
                kind: Attribute,
            },
            CompletionItem {
                label: "windows_subsystem",
                source_range: 20..20,
                delete: 20..20,
                insert: "windows_subsystem = \"${0:subsystem}\"",
                kind: Attribute,
            },
        ]
        "###
        );
    }
}
