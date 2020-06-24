//! Completion for associated items in a trait implementation.
//!
//! This module adds the completion items related to implementing associated
//! items within a `impl Trait for Struct` block. The current context node
//! must be within either a `FN_DEF`, `TYPE_ALIAS_DEF`, or `CONST_DEF` node
//! and an direct child of an `IMPL_DEF`.
//!
//! # Examples
//!
//! Considering the following trait `impl`:
//!
//! ```ignore
//! trait SomeTrait {
//!     fn foo();
//! }
//!
//! impl SomeTrait for () {
//!     fn f<|>
//! }
//! ```
//!
//! may result in the completion of the following method:
//!
//! ```ignore
//! # trait SomeTrait {
//! #    fn foo();
//! # }
//!
//! impl SomeTrait for () {
//!     fn foo() {}<|>
//! }
//! ```

use hir::{self, Docs, HasSource};
use ra_assists::utils::get_missing_assoc_items;
use ra_syntax::{
    ast::{self, edit, ImplDef},
    AstNode, SyntaxKind, SyntaxNode, TextRange, T,
};
use ra_text_edit::TextEdit;

use crate::{
    completion::{
        CompletionContext, CompletionItem, CompletionItemKind, CompletionKind, Completions,
    },
    display::FunctionSignature,
};

pub(crate) fn complete_trait_impl(acc: &mut Completions, ctx: &CompletionContext) {
    if let Some((trigger, impl_def)) = completion_match(ctx) {
        match trigger.kind() {
            SyntaxKind::NAME_REF => get_missing_assoc_items(&ctx.sema, &impl_def)
                .into_iter()
                .for_each(|item| match item {
                    hir::AssocItem::Function(fn_item) => {
                        add_function_impl(&trigger, acc, ctx, fn_item)
                    }
                    hir::AssocItem::TypeAlias(type_item) => {
                        add_type_alias_impl(&trigger, acc, ctx, type_item)
                    }
                    hir::AssocItem::Const(const_item) => {
                        add_const_impl(&trigger, acc, ctx, const_item)
                    }
                }),

            SyntaxKind::FN_DEF => {
                for missing_fn in get_missing_assoc_items(&ctx.sema, &impl_def)
                    .into_iter()
                    .filter_map(|item| match item {
                        hir::AssocItem::Function(fn_item) => Some(fn_item),
                        _ => None,
                    })
                {
                    add_function_impl(&trigger, acc, ctx, missing_fn);
                }
            }

            SyntaxKind::TYPE_ALIAS_DEF => {
                for missing_fn in get_missing_assoc_items(&ctx.sema, &impl_def)
                    .into_iter()
                    .filter_map(|item| match item {
                        hir::AssocItem::TypeAlias(type_item) => Some(type_item),
                        _ => None,
                    })
                {
                    add_type_alias_impl(&trigger, acc, ctx, missing_fn);
                }
            }

            SyntaxKind::CONST_DEF => {
                for missing_fn in get_missing_assoc_items(&ctx.sema, &impl_def)
                    .into_iter()
                    .filter_map(|item| match item {
                        hir::AssocItem::Const(const_item) => Some(const_item),
                        _ => None,
                    })
                {
                    add_const_impl(&trigger, acc, ctx, missing_fn);
                }
            }

            _ => {}
        }
    }
}

fn completion_match(ctx: &CompletionContext) -> Option<(SyntaxNode, ImplDef)> {
    let (trigger, impl_def_offset) = ctx.token.ancestors().find_map(|p| match p.kind() {
        SyntaxKind::FN_DEF
        | SyntaxKind::TYPE_ALIAS_DEF
        | SyntaxKind::CONST_DEF
        | SyntaxKind::BLOCK_EXPR => Some((p, 2)),
        SyntaxKind::NAME_REF => Some((p, 5)),
        _ => None,
    })?;
    let impl_def = (0..impl_def_offset - 1)
        .try_fold(trigger.parent()?, |t, _| t.parent())
        .and_then(ast::ImplDef::cast)?;
    Some((trigger, impl_def))
}

fn add_function_impl(
    fn_def_node: &SyntaxNode,
    acc: &mut Completions,
    ctx: &CompletionContext,
    func: hir::Function,
) {
    let signature = FunctionSignature::from_hir(ctx.db, func);

    let fn_name = func.name(ctx.db).to_string();

    let label = if !func.params(ctx.db).is_empty() {
        format!("fn {}(..)", fn_name)
    } else {
        format!("fn {}()", fn_name)
    };

    let builder = CompletionItem::new(CompletionKind::Magic, ctx.source_range(), label)
        .lookup_by(fn_name)
        .set_documentation(func.docs(ctx.db));

    let completion_kind = if func.has_self_param(ctx.db) {
        CompletionItemKind::Method
    } else {
        CompletionItemKind::Function
    };
    let range = TextRange::new(fn_def_node.text_range().start(), ctx.source_range().end());

    match ctx.config.snippet_cap {
        Some(cap) => {
            let snippet = format!("{} {{\n    $0\n}}", signature);
            builder.snippet_edit(cap, TextEdit::replace(range, snippet))
        }
        None => {
            let header = format!("{} {{", signature);
            builder.text_edit(TextEdit::replace(range, header))
        }
    }
    .kind(completion_kind)
    .add_to(acc);
}

fn add_type_alias_impl(
    type_def_node: &SyntaxNode,
    acc: &mut Completions,
    ctx: &CompletionContext,
    type_alias: hir::TypeAlias,
) {
    let alias_name = type_alias.name(ctx.db).to_string();

    let snippet = format!("type {} = ", alias_name);

    let range = TextRange::new(type_def_node.text_range().start(), ctx.source_range().end());

    CompletionItem::new(CompletionKind::Magic, ctx.source_range(), snippet.clone())
        .text_edit(TextEdit::replace(range, snippet))
        .lookup_by(alias_name)
        .kind(CompletionItemKind::TypeAlias)
        .set_documentation(type_alias.docs(ctx.db))
        .add_to(acc);
}

fn add_const_impl(
    const_def_node: &SyntaxNode,
    acc: &mut Completions,
    ctx: &CompletionContext,
    const_: hir::Const,
) {
    let const_name = const_.name(ctx.db).map(|n| n.to_string());

    if let Some(const_name) = const_name {
        let snippet = make_const_compl_syntax(&const_.source(ctx.db).value);

        let range = TextRange::new(const_def_node.text_range().start(), ctx.source_range().end());

        CompletionItem::new(CompletionKind::Magic, ctx.source_range(), snippet.clone())
            .text_edit(TextEdit::replace(range, snippet))
            .lookup_by(const_name)
            .kind(CompletionItemKind::Const)
            .set_documentation(const_.docs(ctx.db))
            .add_to(acc);
    }
}

fn make_const_compl_syntax(const_: &ast::ConstDef) -> String {
    let const_ = edit::remove_attrs_and_docs(const_);

    let const_start = const_.syntax().text_range().start();
    let const_end = const_.syntax().text_range().end();

    let start =
        const_.syntax().first_child_or_token().map_or(const_start, |f| f.text_range().start());

    let end = const_
        .syntax()
        .children_with_tokens()
        .find(|s| s.kind() == T![;] || s.kind() == T![=])
        .map_or(const_end, |f| f.text_range().start());

    let len = end - start;
    let range = TextRange::new(0.into(), len);

    let syntax = const_.syntax().text().slice(range).to_string();

    format!("{} = ", syntax.trim_end())
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;

    use crate::completion::{test_utils::do_completion, CompletionItem, CompletionKind};

    fn complete(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Magic)
    }

    #[test]
    fn name_ref_function_type_const() {
        let completions = complete(
            r"
            trait Test {
                type TestType;
                const TEST_CONST: u16;
                fn test();
            }

            struct T1;

            impl Test for T1 {
                t<|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "const TEST_CONST: u16 = ",
                source_range: 112..113,
                delete: 112..113,
                insert: "const TEST_CONST: u16 = ",
                kind: Const,
                lookup: "TEST_CONST",
            },
            CompletionItem {
                label: "fn test()",
                source_range: 112..113,
                delete: 112..113,
                insert: "fn test() {\n    $0\n}",
                kind: Function,
                lookup: "test",
            },
            CompletionItem {
                label: "type TestType = ",
                source_range: 112..113,
                delete: 112..113,
                insert: "type TestType = ",
                kind: TypeAlias,
                lookup: "TestType",
            },
        ]
        "###);
    }

    #[test]
    fn no_nested_fn_completions() {
        let completions = complete(
            r"
            trait Test {
                fn test();
                fn test2();
            }

            struct T1;

            impl Test for T1 {
                fn test() {
                    t<|>
                }
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"[]"###);
    }

    #[test]
    fn name_ref_single_function() {
        let completions = complete(
            r"
            trait Test {
                fn test();
            }

            struct T1;

            impl Test for T1 {
                t<|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "fn test()",
                source_range: 66..67,
                delete: 66..67,
                insert: "fn test() {\n    $0\n}",
                kind: Function,
                lookup: "test",
            },
        ]
        "###);
    }

    #[test]
    fn single_function() {
        let completions = complete(
            r"
            trait Test {
                fn foo();
            }

            struct T1;

            impl Test for T1 {
                fn f<|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "fn foo()",
                source_range: 68..69,
                delete: 65..69,
                insert: "fn foo() {\n    $0\n}",
                kind: Function,
                lookup: "foo",
            },
        ]
        "###);
    }

    #[test]
    fn hide_implemented_fn() {
        let completions = complete(
            r"
            trait Test {
                fn foo();
                fn foo_bar();
            }

            struct T1;

            impl Test for T1 {
                fn foo() {}

                fn f<|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "fn foo_bar()",
                source_range: 103..104,
                delete: 100..104,
                insert: "fn foo_bar() {\n    $0\n}",
                kind: Function,
                lookup: "foo_bar",
            },
        ]
        "###);
    }

    #[test]
    fn completes_only_on_top_level() {
        let completions = complete(
            r"
            trait Test {
                fn foo();

                fn foo_bar();
            }

            struct T1;

            impl Test for T1 {
                fn foo() {
                    <|>
                }
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"[]"###);
    }

    #[test]
    fn generic_fn() {
        let completions = complete(
            r"
            trait Test {
                fn foo<T>();
            }

            struct T1;

            impl Test for T1 {
                fn f<|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "fn foo()",
                source_range: 71..72,
                delete: 68..72,
                insert: "fn foo<T>() {\n    $0\n}",
                kind: Function,
                lookup: "foo",
            },
        ]
        "###);
    }

    #[test]
    fn generic_constrait_fn() {
        let completions = complete(
            r"
            trait Test {
                fn foo<T>() where T: Into<String>;
            }

            struct T1;

            impl Test for T1 {
                fn f<|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "fn foo()",
                source_range: 93..94,
                delete: 90..94,
                insert: "fn foo<T>()\nwhere T: Into<String> {\n    $0\n}",
                kind: Function,
                lookup: "foo",
            },
        ]
        "###);
    }

    #[test]
    fn associated_type() {
        let completions = complete(
            r"
            trait Test {
                type SomeType;
            }

            impl Test for () {
                type S<|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "type SomeType = ",
                source_range: 63..64,
                delete: 58..64,
                insert: "type SomeType = ",
                kind: TypeAlias,
                lookup: "SomeType",
            },
        ]
        "###);
    }

    #[test]
    fn associated_const() {
        let completions = complete(
            r"
            trait Test {
                const SOME_CONST: u16;
            }

            impl Test for () {
                const S<|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "const SOME_CONST: u16 = ",
                source_range: 72..73,
                delete: 66..73,
                insert: "const SOME_CONST: u16 = ",
                kind: Const,
                lookup: "SOME_CONST",
            },
        ]
        "###);
    }

    #[test]
    fn associated_const_with_default() {
        let completions = complete(
            r"
            trait Test {
                const SOME_CONST: u16 = 42;
            }

            impl Test for () {
                const S<|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "const SOME_CONST: u16 = ",
                source_range: 77..78,
                delete: 71..78,
                insert: "const SOME_CONST: u16 = ",
                kind: Const,
                lookup: "SOME_CONST",
            },
        ]
        "###);
    }
}
