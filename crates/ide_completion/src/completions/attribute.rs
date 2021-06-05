//! Completion for attributes
//!
//! This module uses a bit of static metadata to provide completions
//! for built-in attributes.

use hir::HasAttrs;
use ide_db::helpers::generated_lints::{CLIPPY_LINTS, DEFAULT_LINTS, FEATURES};
use once_cell::sync::Lazy;
use rustc_hash::{FxHashMap, FxHashSet};
use syntax::{algo::non_trivia_sibling, ast, AstNode, Direction, NodeOrToken, SyntaxKind, T};

use crate::{
    context::CompletionContext,
    item::{CompletionItem, CompletionItemKind, CompletionKind},
    Completions,
};

mod derive;
mod lint;

pub(crate) fn complete_attribute(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    let attribute = ctx.attribute_under_caret.as_ref()?;
    match (attribute.path().and_then(|p| p.as_single_name_ref()), attribute.token_tree()) {
        (Some(path), Some(token_tree)) => match path.text().as_str() {
            "derive" => derive::complete_derive(acc, ctx, token_tree),
            "feature" => lint::complete_lint(acc, ctx, token_tree, FEATURES),
            "allow" | "warn" | "deny" | "forbid" => {
                lint::complete_lint(acc, ctx, token_tree.clone(), DEFAULT_LINTS);
                lint::complete_lint(acc, ctx, token_tree, CLIPPY_LINTS);
            }
            _ => (),
        },
        (None, Some(_)) => (),
        _ => complete_new_attribute(acc, ctx, attribute),
    }
    Some(())
}

fn complete_new_attribute(acc: &mut Completions, ctx: &CompletionContext, attribute: &ast::Attr) {
    let is_inner = attribute.kind() == ast::AttrKind::Inner;
    let attribute_annotated_item_kind =
        attribute.syntax().parent().map(|it| it.kind()).filter(|_| {
            is_inner
            // If we got nothing coming after the attribute it could be anything so filter it the kind out
                || non_trivia_sibling(attribute.syntax().clone().into(), Direction::Next).is_some()
        });
    let attributes = attribute_annotated_item_kind.and_then(|kind| {
        if ast::Expr::can_cast(kind) {
            Some(EXPR_ATTRIBUTES)
        } else {
            KIND_TO_ATTRIBUTES.get(&kind).copied()
        }
    });

    let add_completion = |attr_completion: &AttrCompletion| {
        let mut item = CompletionItem::new(
            CompletionKind::Attribute,
            ctx.source_range(),
            attr_completion.label,
        );
        item.kind(CompletionItemKind::Attribute);

        if let Some(lookup) = attr_completion.lookup {
            item.lookup_by(lookup);
        }

        if let Some((snippet, cap)) = attr_completion.snippet.zip(ctx.config.snippet_cap) {
            item.insert_snippet(cap, snippet);
        }

        if is_inner || !attr_completion.prefer_inner {
            acc.add(item.build());
        }
    };

    match attributes {
        Some(applicable) => applicable
            .iter()
            .flat_map(|name| ATTRIBUTES.binary_search_by(|attr| attr.key().cmp(name)).ok())
            .flat_map(|idx| ATTRIBUTES.get(idx))
            .for_each(add_completion),
        None if is_inner => ATTRIBUTES.iter().for_each(add_completion),
        None => ATTRIBUTES.iter().filter(|compl| !compl.prefer_inner).for_each(add_completion),
    }

    // FIXME: write a test for this when we can
    ctx.scope.process_all_names(&mut |name, scope_def| {
        if let hir::ScopeDef::MacroDef(mac) = scope_def {
            if mac.kind() == hir::MacroKind::Attr {
                let mut item = CompletionItem::new(
                    CompletionKind::Attribute,
                    ctx.source_range(),
                    name.to_string(),
                );
                item.kind(CompletionItemKind::Attribute);
                if let Some(docs) = mac.docs(ctx.sema.db) {
                    item.documentation(docs);
                }
                acc.add(item.build());
            }
        }
    });
}

struct AttrCompletion {
    label: &'static str,
    lookup: Option<&'static str>,
    snippet: Option<&'static str>,
    prefer_inner: bool,
}

impl AttrCompletion {
    fn key(&self) -> &'static str {
        self.lookup.unwrap_or(self.label)
    }

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

macro_rules! attrs {
    // attributes applicable to all items
    [@ { item $($tt:tt)* } {$($acc:tt)*}] => {
        attrs!(@ { $($tt)* } { $($acc)*, "deprecated", "doc", "dochidden", "docalias", "must_use", "no_mangle" })
    };
    // attributes applicable to all adts
    [@ { adt $($tt:tt)* } {$($acc:tt)*}] => {
        attrs!(@ { $($tt)* } { $($acc)*, "derive", "repr" })
    };
    // attributes applicable to all linkable things aka functions/statics
    [@ { linkable $($tt:tt)* } {$($acc:tt)*}] => {
        attrs!(@ { $($tt)* } { $($acc)*, "export_name", "link_name", "link_section" })
    };
    // error fallback for nicer error message
    [@ { $ty:ident $($tt:tt)* } {$($acc:tt)*}] => {
        compile_error!(concat!("unknown attr subtype ", stringify!($ty)))
    };
    // general push down accumulation
    [@ { $lit:literal $($tt:tt)*} {$($acc:tt)*}] => {
        attrs!(@ { $($tt)* } { $($acc)*, $lit })
    };
    [@ {$($tt:tt)+} {$($tt2:tt)*}] => {
        compile_error!(concat!("Unexpected input ", stringify!($($tt)+)))
    };
    // final output construction
    [@ {} {$($tt:tt)*}] => { &[$($tt)*] as _ };
    // starting matcher
    [$($tt:tt),*] => {
        attrs!(@ { $($tt)* } { "allow", "cfg", "cfg_attr", "deny", "forbid", "warn" })
    };
}

#[rustfmt::skip]
static KIND_TO_ATTRIBUTES: Lazy<FxHashMap<SyntaxKind, &[&str]>> = Lazy::new(|| {
    use SyntaxKind::*;
    std::array::IntoIter::new([
        (
            SOURCE_FILE,
            attrs!(
                item,
                "crate_name", "feature", "no_implicit_prelude", "no_main", "no_std",
                "recursion_limit", "type_length_limit", "windows_subsystem"
            ),
        ),
        (MODULE, attrs!(item, "no_implicit_prelude", "path")),
        (ITEM_LIST, attrs!(item, "no_implicit_prelude")),
        (MACRO_RULES, attrs!(item, "macro_export", "macro_use")),
        (MACRO_DEF, attrs!(item)),
        (EXTERN_CRATE, attrs!(item, "macro_use", "no_link")),
        (USE, attrs!(item)),
        (TYPE_ALIAS, attrs!(item)),
        (STRUCT, attrs!(item, adt, "non_exhaustive")),
        (ENUM, attrs!(item, adt, "non_exhaustive")),
        (UNION, attrs!(item, adt)),
        (CONST, attrs!(item)),
        (
            FN,
            attrs!(
                item, linkable,
                "cold", "ignore", "inline", "must_use", "panic_handler", "proc_macro",
                "proc_macro_derive", "proc_macro_attribute", "should_panic", "target_feature",
                "test", "track_caller"
            ),
        ),
        (STATIC, attrs!(item, linkable, "global_allocator", "used")),
        (TRAIT, attrs!(item, "must_use")),
        (IMPL, attrs!(item, "automatically_derived")),
        (ASSOC_ITEM_LIST, attrs!(item)),
        (EXTERN_BLOCK, attrs!(item, "link")),
        (EXTERN_ITEM_LIST, attrs!(item, "link")),
        (MACRO_CALL, attrs!()),
        (SELF_PARAM, attrs!()),
        (PARAM, attrs!()),
        (RECORD_FIELD, attrs!()),
        (VARIANT, attrs!("non_exhaustive")),
        (TYPE_PARAM, attrs!()),
        (CONST_PARAM, attrs!()),
        (LIFETIME_PARAM, attrs!()),
        (LET_STMT, attrs!()),
        (EXPR_STMT, attrs!()),
        (LITERAL, attrs!()),
        (RECORD_EXPR_FIELD_LIST, attrs!()),
        (RECORD_EXPR_FIELD, attrs!()),
        (MATCH_ARM_LIST, attrs!()),
        (MATCH_ARM, attrs!()),
        (IDENT_PAT, attrs!()),
        (RECORD_PAT_FIELD, attrs!()),
    ])
    .collect()
});
const EXPR_ATTRIBUTES: &[&str] = attrs!();

/// https://doc.rust-lang.org/reference/attributes.html#built-in-attributes-index
// Keep these sorted for the binary search!
const ATTRIBUTES: &[AttrCompletion] = &[
    attr("allow(…)", Some("allow"), Some("allow(${0:lint})")),
    attr("automatically_derived", None, None),
    attr("cfg(…)", Some("cfg"), Some("cfg(${0:predicate})")),
    attr("cfg_attr(…)", Some("cfg_attr"), Some("cfg_attr(${1:predicate}, ${0:attr})")),
    attr("cold", None, None),
    attr(r#"crate_name = """#, Some("crate_name"), Some(r#"crate_name = "${0:crate_name}""#))
        .prefer_inner(),
    attr("deny(…)", Some("deny"), Some("deny(${0:lint})")),
    attr(r#"deprecated"#, Some("deprecated"), Some(r#"deprecated"#)),
    attr("derive(…)", Some("derive"), Some(r#"derive(${0:Debug})"#)),
    attr(r#"doc = "…""#, Some("doc"), Some(r#"doc = "${0:docs}""#)),
    attr(r#"doc(alias = "…")"#, Some("docalias"), Some(r#"doc(alias = "${0:docs}")"#)),
    attr(r#"doc(hidden)"#, Some("dochidden"), Some(r#"doc(hidden)"#)),
    attr(
        r#"export_name = "…""#,
        Some("export_name"),
        Some(r#"export_name = "${0:exported_symbol_name}""#),
    ),
    attr("feature(…)", Some("feature"), Some("feature(${0:flag})")).prefer_inner(),
    attr("forbid(…)", Some("forbid"), Some("forbid(${0:lint})")),
    attr("global_allocator", None, None),
    attr(r#"ignore = "…""#, Some("ignore"), Some(r#"ignore = "${0:reason}""#)),
    attr("inline", Some("inline"), Some("inline")),
    attr("link", None, None),
    attr(r#"link_name = "…""#, Some("link_name"), Some(r#"link_name = "${0:symbol_name}""#)),
    attr(
        r#"link_section = "…""#,
        Some("link_section"),
        Some(r#"link_section = "${0:section_name}""#),
    ),
    attr("macro_export", None, None),
    attr("macro_use", None, None),
    attr(r#"must_use"#, Some("must_use"), Some(r#"must_use"#)),
    attr("no_implicit_prelude", None, None).prefer_inner(),
    attr("no_link", None, None).prefer_inner(),
    attr("no_main", None, None).prefer_inner(),
    attr("no_mangle", None, None),
    attr("no_std", None, None).prefer_inner(),
    attr("non_exhaustive", None, None),
    attr("panic_handler", None, None),
    attr(r#"path = "…""#, Some("path"), Some(r#"path ="${0:path}""#)),
    attr("proc_macro", None, None),
    attr("proc_macro_attribute", None, None),
    attr("proc_macro_derive(…)", Some("proc_macro_derive"), Some("proc_macro_derive(${0:Trait})")),
    attr("recursion_limit = …", Some("recursion_limit"), Some("recursion_limit = ${0:128}"))
        .prefer_inner(),
    attr("repr(…)", Some("repr"), Some("repr(${0:C})")),
    attr("should_panic", Some("should_panic"), Some(r#"should_panic"#)),
    attr(
        r#"target_feature = "…""#,
        Some("target_feature"),
        Some(r#"target_feature = "${0:feature}""#),
    ),
    attr("test", None, None),
    attr("track_caller", None, None),
    attr("type_length_limit = …", Some("type_length_limit"), Some("type_length_limit = ${0:128}"))
        .prefer_inner(),
    attr("used", None, None),
    attr("warn(…)", Some("warn"), Some("warn(${0:lint})")),
    attr(
        r#"windows_subsystem = "…""#,
        Some("windows_subsystem"),
        Some(r#"windows_subsystem = "${0:subsystem}""#),
    )
    .prefer_inner(),
];

fn parse_comma_sep_input(derive_input: ast::TokenTree) -> Option<FxHashSet<String>> {
    let (l_paren, r_paren) = derive_input.l_paren_token().zip(derive_input.r_paren_token())?;
    let mut input_derives = FxHashSet::default();
    let mut tokens = derive_input
        .syntax()
        .children_with_tokens()
        .filter_map(NodeOrToken::into_token)
        .skip_while(|token| token != &l_paren)
        .skip(1)
        .take_while(|token| token != &r_paren)
        .peekable();
    let mut input = String::new();
    while tokens.peek().is_some() {
        for token in tokens.by_ref().take_while(|t| t.kind() != T![,]) {
            input.push_str(token.text());
        }

        if !input.is_empty() {
            input_derives.insert(input.trim().to_owned());
        }

        input.clear();
    }

    Some(input_derives)
}

#[cfg(test)]
mod tests {
    use super::*;

    use expect_test::{expect, Expect};

    use crate::{test_utils::completion_list, CompletionKind};

    #[test]
    fn attributes_are_sorted() {
        let mut attrs = ATTRIBUTES.iter().map(|attr| attr.key());
        let mut prev = attrs.next().unwrap();

        attrs.for_each(|next| {
            assert!(
                prev < next,
                r#"ATTRIBUTES array is not sorted, "{}" should come after "{}""#,
                prev,
                next
            );
            prev = next;
        });
    }

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Attribute);
        expect.assert_eq(&actual);
    }

    #[test]
    fn test_attribute_completion_inside_nested_attr() {
        check(r#"#[cfg($0)]"#, expect![[]])
    }

    #[test]
    fn test_attribute_completion_with_existing_attr() {
        check(
            r#"#[no_mangle] #[$0] mcall!();"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
            "#]],
        )
    }

    #[test]
    fn complete_attribute_on_source_file() {
        check(
            r#"#![$0]"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
                at crate_name = ""
                at feature(…)
                at no_implicit_prelude
                at no_main
                at no_std
                at recursion_limit = …
                at type_length_limit = …
                at windows_subsystem = "…"
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_module() {
        check(
            r#"#[$0] mod foo;"#,
            expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at forbid(…)
            at warn(…)
            at deprecated
            at doc = "…"
            at doc(hidden)
            at doc(alias = "…")
            at must_use
            at no_mangle
            at path = "…"
        "#]],
        );
        check(
            r#"mod foo {#![$0]}"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
                at no_implicit_prelude
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_macro_rules() {
        check(
            r#"#[$0] macro_rules! foo {}"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
                at macro_export
                at macro_use
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_macro_def() {
        check(
            r#"#[$0] macro foo {}"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_extern_crate() {
        check(
            r#"#[$0] extern crate foo;"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
                at macro_use
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_use() {
        check(
            r#"#[$0] use foo;"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_type_alias() {
        check(
            r#"#[$0] type foo = ();"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_struct() {
        check(
            r#"#[$0] struct Foo;"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
                at derive(…)
                at repr(…)
                at non_exhaustive
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_enum() {
        check(
            r#"#[$0] enum Foo {}"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
                at derive(…)
                at repr(…)
                at non_exhaustive
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_const() {
        check(
            r#"#[$0] const FOO: () = ();"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_static() {
        check(
            r#"#[$0] static FOO: () = ()"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
                at export_name = "…"
                at link_name = "…"
                at link_section = "…"
                at global_allocator
                at used
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_trait() {
        check(
            r#"#[$0] trait Foo {}"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
                at must_use
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_impl() {
        check(
            r#"#[$0] impl () {}"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
                at automatically_derived
            "#]],
        );
        check(
            r#"impl () {#![$0]}"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_extern_block() {
        check(
            r#"#[$0] extern {}"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
                at link
            "#]],
        );
        check(
            r#"extern {#![$0]}"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
                at link
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_variant() {
        check(
            r#"enum Foo { #[$0] Bar }"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at non_exhaustive
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_fn() {
        check(
            r#"#[$0] fn main() {}"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
                at deprecated
                at doc = "…"
                at doc(hidden)
                at doc(alias = "…")
                at must_use
                at no_mangle
                at export_name = "…"
                at link_name = "…"
                at link_section = "…"
                at cold
                at ignore = "…"
                at inline
                at must_use
                at panic_handler
                at proc_macro
                at proc_macro_derive(…)
                at proc_macro_attribute
                at should_panic
                at target_feature = "…"
                at test
                at track_caller
            "#]],
        );
    }

    #[test]
    fn complete_attribute_on_expr() {
        check(
            r#"fn main() { #[$0] foo() }"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at forbid(…)
                at warn(…)
            "#]],
        );
    }

    #[test]
    fn complete_attribute_in_source_file_end() {
        check(
            r#"#[$0]"#,
            expect![[r#"
                at allow(…)
                at automatically_derived
                at cfg(…)
                at cfg_attr(…)
                at cold
                at deny(…)
                at deprecated
                at derive(…)
                at doc = "…"
                at doc(alias = "…")
                at doc(hidden)
                at export_name = "…"
                at forbid(…)
                at global_allocator
                at ignore = "…"
                at inline
                at link
                at link_name = "…"
                at link_section = "…"
                at macro_export
                at macro_use
                at must_use
                at no_mangle
                at non_exhaustive
                at panic_handler
                at path = "…"
                at proc_macro
                at proc_macro_attribute
                at proc_macro_derive(…)
                at repr(…)
                at should_panic
                at target_feature = "…"
                at test
                at track_caller
                at used
                at warn(…)
            "#]],
        );
    }
}
