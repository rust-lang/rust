//! Completion for (built-in) attributes, derives and lints.
//!
//! This module uses a bit of static metadata to provide completions for builtin-in attributes and lints.

use std::sync::LazyLock;

use ide_db::{
    FxHashMap, SymbolKind,
    generated::lints::{
        CLIPPY_LINT_GROUPS, CLIPPY_LINTS, DEFAULT_LINTS, FEATURES, Lint, RUSTDOC_LINTS,
    },
    syntax_helpers::node_ext::parse_tt_as_comma_sep_paths,
};
use itertools::Itertools;
use syntax::{
    AstNode, Edition, SyntaxKind, T,
    ast::{self, AttrKind},
};

use crate::{
    Completions,
    context::{AttrCtx, CompletionContext, PathCompletionCtx, Qualified},
    item::CompletionItem,
};

mod cfg;
mod derive;
mod diagnostic;
mod lint;
mod macro_use;
mod repr;

pub(crate) use self::derive::complete_derive_path;

/// Complete inputs to known builtin attributes as well as derive attributes
pub(crate) fn complete_known_attribute_input(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    &colon_prefix: &bool,
    fake_attribute_under_caret: &ast::Attr,
    extern_crate: Option<&ast::ExternCrate>,
) -> Option<()> {
    let attribute = fake_attribute_under_caret;
    let path = attribute.path()?;
    let segments = path.segments().map(|s| s.name_ref()).collect::<Option<Vec<_>>>()?;
    let segments = segments.iter().map(|n| n.text()).collect::<Vec<_>>();
    let segments = segments.iter().map(|t| t.as_str()).collect::<Vec<_>>();
    let tt = attribute.token_tree()?;

    match segments.as_slice() {
        ["repr"] => repr::complete_repr(acc, ctx, tt),
        ["feature"] => lint::complete_lint(
            acc,
            ctx,
            colon_prefix,
            &parse_tt_as_comma_sep_paths(tt, ctx.edition)?,
            FEATURES,
        ),
        ["allow"] | ["expect"] | ["deny"] | ["forbid"] | ["warn"] => {
            let existing_lints = parse_tt_as_comma_sep_paths(tt, ctx.edition)?;

            let lints: Vec<Lint> = CLIPPY_LINT_GROUPS
                .iter()
                .map(|g| &g.lint)
                .chain(DEFAULT_LINTS)
                .chain(CLIPPY_LINTS)
                .chain(RUSTDOC_LINTS)
                .cloned()
                .collect();

            lint::complete_lint(acc, ctx, colon_prefix, &existing_lints, &lints);
        }
        ["cfg"] => cfg::complete_cfg(acc, ctx),
        ["macro_use"] => macro_use::complete_macro_use(
            acc,
            ctx,
            extern_crate,
            &parse_tt_as_comma_sep_paths(tt, ctx.edition)?,
        ),
        ["diagnostic", "on_unimplemented"] => diagnostic::complete_on_unimplemented(acc, ctx, tt),
        _ => (),
    }
    Some(())
}

pub(crate) fn complete_attribute_path(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx @ PathCompletionCtx { qualified, .. }: &PathCompletionCtx<'_>,
    &AttrCtx { kind, annotated_item_kind, ref derive_helpers }: &AttrCtx,
) {
    let is_inner = kind == AttrKind::Inner;

    for (derive_helper, derive_name) in derive_helpers {
        let mut item = CompletionItem::new(
            SymbolKind::Attribute,
            ctx.source_range(),
            derive_helper.as_str(),
            ctx.edition,
        );
        item.detail(format!("derive helper of `{derive_name}`"));
        item.add_to(acc, ctx.db);
    }

    match qualified {
        Qualified::With {
            resolution: Some(hir::PathResolution::Def(hir::ModuleDef::Module(module))),
            super_chain_len,
            ..
        } => {
            acc.add_super_keyword(ctx, *super_chain_len);

            for (name, def) in module.scope(ctx.db, Some(ctx.module)) {
                match def {
                    hir::ScopeDef::ModuleDef(hir::ModuleDef::Macro(m)) if m.is_attr(ctx.db) => {
                        acc.add_macro(ctx, path_ctx, m, name)
                    }
                    hir::ScopeDef::ModuleDef(hir::ModuleDef::Module(m)) => {
                        acc.add_module(ctx, path_ctx, m, name, vec![])
                    }
                    _ => (),
                }
            }
            return;
        }
        // fresh use tree with leading colon2, only show crate roots
        Qualified::Absolute => acc.add_crate_roots(ctx, path_ctx),
        // only show modules in a fresh UseTree
        Qualified::No => {
            ctx.process_all_names(&mut |name, def, doc_aliases| match def {
                hir::ScopeDef::ModuleDef(hir::ModuleDef::Macro(m)) if m.is_attr(ctx.db) => {
                    acc.add_macro(ctx, path_ctx, m, name)
                }
                hir::ScopeDef::ModuleDef(hir::ModuleDef::Module(m)) => {
                    acc.add_module(ctx, path_ctx, m, name, doc_aliases)
                }
                _ => (),
            });
            acc.add_nameref_keywords_with_colon(ctx);
        }
        Qualified::TypeAnchor { .. } | Qualified::With { .. } => {}
    }
    let qualifier_path =
        if let Qualified::With { path, .. } = qualified { Some(path) } else { None };

    let attributes = annotated_item_kind.and_then(|kind| {
        if ast::Expr::can_cast(kind) {
            Some(EXPR_ATTRIBUTES)
        } else {
            KIND_TO_ATTRIBUTES.get(&kind).copied()
        }
    });

    let add_completion = |attr_completion: &AttrCompletion| {
        // if we don't already have the qualifiers of the completion, then
        // add the missing parts to the label and snippet
        let mut label = attr_completion.label.to_owned();
        let mut snippet = attr_completion.snippet.map(|s| s.to_owned());
        let segments = qualifier_path.iter().flat_map(|q| q.segments()).collect::<Vec<_>>();
        let qualifiers = attr_completion.qualifiers;
        let matching_qualifiers = segments
            .iter()
            .zip(qualifiers)
            .take_while(|(s, q)| s.name_ref().is_some_and(|t| t.text() == **q))
            .count();
        if matching_qualifiers != qualifiers.len() {
            let prefix = qualifiers[matching_qualifiers..].join("::");
            label = format!("{prefix}::{label}");
            if let Some(s) = snippet.as_mut() {
                *s = format!("{prefix}::{s}");
            }
        }

        let mut item =
            CompletionItem::new(SymbolKind::Attribute, ctx.source_range(), label, ctx.edition);

        if let Some(lookup) = attr_completion.lookup {
            item.lookup_by(lookup);
        }

        if let Some((snippet, cap)) = snippet.zip(ctx.config.snippet_cap) {
            item.insert_snippet(cap, snippet);
        }

        if is_inner || !attr_completion.prefer_inner {
            item.add_to(acc, ctx.db);
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
}

struct AttrCompletion {
    label: &'static str,
    lookup: Option<&'static str>,
    snippet: Option<&'static str>,
    qualifiers: &'static [&'static str],
    prefer_inner: bool,
}

impl AttrCompletion {
    fn key(&self) -> &'static str {
        self.lookup.unwrap_or(self.label)
    }

    const fn qualifiers(self, qualifiers: &'static [&'static str]) -> AttrCompletion {
        AttrCompletion { qualifiers, ..self }
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
    AttrCompletion { label, lookup, snippet, qualifiers: &[], prefer_inner: false }
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
        attrs!(@ { $($tt)* } { "allow", "cfg", "cfg_attr", "deny", "expect", "forbid", "warn" })
    };
}

#[rustfmt::skip]
static KIND_TO_ATTRIBUTES: LazyLock<FxHashMap<SyntaxKind, &[&str]>> = LazyLock::new(|| {
    use SyntaxKind::*;
    [
        (
            SOURCE_FILE,
            attrs!(
                item,
                "crate_name", "feature", "no_implicit_prelude", "no_main", "no_std",
                "recursion_limit", "type_length_limit", "windows_subsystem"
            ),
        ),
        (MODULE, attrs!(item, "macro_use", "no_implicit_prelude", "path")),
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
                "cold", "ignore", "inline", "panic_handler", "proc_macro",
                "proc_macro_derive", "proc_macro_attribute", "should_panic", "target_feature",
                "test", "track_caller"
            ),
        ),
        (STATIC, attrs!(item, linkable, "global_allocator", "used")),
        (TRAIT, attrs!(item, "diagnostic::on_unimplemented")),
        (IMPL, attrs!(item, "automatically_derived", "diagnostic::do_not_recommend")),
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
    ]
    .into_iter()
    .collect()
});
const EXPR_ATTRIBUTES: &[&str] = attrs!();

/// <https://doc.rust-lang.org/reference/attributes.html#built-in-attributes-index>
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
    attr("do_not_recommend", Some("diagnostic::do_not_recommend"), None)
        .qualifiers(&["diagnostic"]),
    attr(
        "on_unimplemented",
        Some("diagnostic::on_unimplemented"),
        Some(r#"on_unimplemented(${0:keys})"#),
    )
    .qualifiers(&["diagnostic"]),
    attr(r#"doc = "…""#, Some("doc"), Some(r#"doc = "${0:docs}""#)),
    attr(r#"doc(alias = "…")"#, Some("docalias"), Some(r#"doc(alias = "${0:docs}")"#)),
    attr(r#"doc(hidden)"#, Some("dochidden"), Some(r#"doc(hidden)"#)),
    attr("expect(…)", Some("expect"), Some("expect(${0:lint})")),
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
    attr(
        r#"recursion_limit = "…""#,
        Some("recursion_limit"),
        Some(r#"recursion_limit = "${0:128}""#),
    )
    .prefer_inner(),
    attr("repr(…)", Some("repr"), Some("repr(${0:C})")),
    attr("should_panic", Some("should_panic"), Some(r#"should_panic"#)),
    attr(
        r#"target_feature(enable = "…")"#,
        Some("target_feature"),
        Some(r#"target_feature(enable = "${0:feature}")"#),
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

fn parse_comma_sep_expr(input: ast::TokenTree) -> Option<Vec<ast::Expr>> {
    let r_paren = input.r_paren_token()?;
    let tokens = input
        .syntax()
        .children_with_tokens()
        .skip(1)
        .take_while(|it| it.as_token() != Some(&r_paren));
    let input_expressions = tokens.chunk_by(|tok| tok.kind() == T![,]);
    Some(
        input_expressions
            .into_iter()
            .filter_map(|(is_sep, group)| (!is_sep).then_some(group))
            .filter_map(|mut tokens| {
                syntax::hacks::parse_expr_from_str(&tokens.join(""), Edition::CURRENT)
            })
            .collect::<Vec<ast::Expr>>(),
    )
}

#[test]
fn attributes_are_sorted() {
    let mut attrs = ATTRIBUTES.iter().map(|attr| attr.key());
    let mut prev = attrs.next().unwrap();

    attrs.for_each(|next| {
        assert!(
            prev < next,
            r#"ATTRIBUTES array is not sorted, "{prev}" should come after "{next}""#
        );
        prev = next;
    });
}
