//! FIXME: write short doc here

use crate::{
    completion::{
        CompletionContext, CompletionItem, CompletionItemKind, CompletionKind, Completions,
    },
    display::FunctionSignature,
};

use hir::{self, Docs, HasSource};
use ra_syntax::{
    ast::{self, edit},
    AstNode, SyntaxKind, TextRange,
};

use ra_assists::utils::get_missing_impl_items;

pub(crate) fn complete_trait_impl(acc: &mut Completions, ctx: &CompletionContext) {
    let trigger = ctx.token.ancestors().find(|p| match p.kind() {
        SyntaxKind::FN_DEF | SyntaxKind::TYPE_ALIAS_DEF | SyntaxKind::CONST_DEF => true,
        _ => false,
    });

    let impl_block = trigger
        .as_ref()
        .and_then(|node| node.parent())
        .and_then(|node| node.parent())
        .and_then(|node| ast::ImplBlock::cast(node));

    if let (Some(trigger), Some(impl_block)) = (trigger, impl_block) {
        match trigger.kind() {
            SyntaxKind::FN_DEF => {
                for missing_fn in get_missing_impl_items(ctx.db, &ctx.analyzer, &impl_block)
                    .iter()
                    .filter_map(|item| match item {
                        hir::AssocItem::Function(fn_item) => Some(fn_item),
                        _ => None,
                    })
                {
                    add_function_impl(acc, ctx, &missing_fn);
                }
            }

            SyntaxKind::TYPE_ALIAS_DEF => {
                for missing_fn in get_missing_impl_items(ctx.db, &ctx.analyzer, &impl_block)
                    .iter()
                    .filter_map(|item| match item {
                        hir::AssocItem::TypeAlias(type_item) => Some(type_item),
                        _ => None,
                    })
                {
                    add_type_alias_impl(acc, ctx, &missing_fn);
                }
            }

            SyntaxKind::CONST_DEF => {
                for missing_fn in get_missing_impl_items(ctx.db, &ctx.analyzer, &impl_block)
                    .iter()
                    .filter_map(|item| match item {
                        hir::AssocItem::Const(const_item) => Some(const_item),
                        _ => None,
                    })
                {
                    add_const_impl(acc, ctx, &missing_fn);
                }
            }

            _ => {}
        }
    }
}

fn add_function_impl(acc: &mut Completions, ctx: &CompletionContext, func: &hir::Function) {
    let display = FunctionSignature::from_hir(ctx.db, func.clone());

    let func_name = func.name(ctx.db);

    let label = if func.params(ctx.db).len() > 0 {
        format!("fn {}(..)", func_name.to_string())
    } else {
        format!("fn {}()", func_name.to_string())
    };

    let builder = CompletionItem::new(CompletionKind::Magic, ctx.source_range(), label.clone())
        .lookup_by(label)
        .set_documentation(func.docs(ctx.db));

    let completion_kind = if func.has_self_param(ctx.db) {
        CompletionItemKind::Method
    } else {
        CompletionItemKind::Function
    };

    let snippet = format!("{} {{}}", display);

    builder.insert_text(snippet).kind(completion_kind).add_to(acc);
}

fn add_type_alias_impl(
    acc: &mut Completions,
    ctx: &CompletionContext,
    type_alias: &hir::TypeAlias,
) {
    let snippet = format!("type {} = ", type_alias.name(ctx.db).to_string());

    CompletionItem::new(CompletionKind::Magic, ctx.source_range(), snippet.clone())
        .insert_text(snippet)
        .kind(CompletionItemKind::TypeAlias)
        .set_documentation(type_alias.docs(ctx.db))
        .add_to(acc);
}

fn add_const_impl(acc: &mut Completions, ctx: &CompletionContext, const_: &hir::Const) {
    let snippet = make_const_compl_syntax(&const_.source(ctx.db).value);

    CompletionItem::new(CompletionKind::Magic, ctx.source_range(), snippet.clone())
        .insert_text(snippet)
        .kind(CompletionItemKind::Const)
        .set_documentation(const_.docs(ctx.db))
        .add_to(acc);
}

fn make_const_compl_syntax(const_: &ast::ConstDef) -> String {
    let const_ = edit::strip_attrs_and_docs(const_);

    let const_start = const_.syntax().text_range().start();
    let const_end = const_.syntax().text_range().end();

    let start =
        const_.syntax().first_child_or_token().map_or(const_start, |f| f.text_range().start());

    let end = const_
        .syntax()
        .children_with_tokens()
        .find(|s| s.kind() == SyntaxKind::SEMI || s.kind() == SyntaxKind::EQ)
        .map_or(const_end, |f| f.text_range().start());

    let len = end - start;
    let range = TextRange::from_to(0.into(), len);

    let syntax = const_.syntax().text().slice(range).to_string();

    format!("{} = ", syntax.trim_end())
}

#[cfg(test)]
mod tests {
    use crate::completion::{do_completion, CompletionItem, CompletionKind};
    use insta::assert_debug_snapshot;

    fn complete(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Magic)
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
                fn<|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "fn foo()",
                source_range: [140; 140),
                delete: [140; 140),
                insert: "fn foo() {}",
                kind: Function,
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
                fn bar();
            }

            struct T1;

            impl Test for T1 {
                fn foo() {}

                fn<|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "fn bar()",
                source_range: [195; 195),
                delete: [195; 195),
                insert: "fn bar() {}",
                kind: Function,
            },
        ]
        "###);
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
                fn<|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "fn foo()",
                source_range: [143; 143),
                delete: [143; 143),
                insert: "fn foo<T>() {}",
                kind: Function,
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
                fn<|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "fn foo()",
                source_range: [165; 165),
                delete: [165; 165),
                insert: "fn foo<T>()\nwhere T: Into<String> {}",
                kind: Function,
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
                type<|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "type SomeType = ",
                source_range: [123; 123),
                delete: [123; 123),
                insert: "type SomeType = ",
                kind: TypeAlias,
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
                source_range: [133; 134),
                delete: [133; 134),
                insert: "const SOME_CONST: u16 = ",
                kind: Const,
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
                source_range: [138; 139),
                delete: [138; 139),
                insert: "const SOME_CONST: u16 = ",
                kind: Const,
            },
        ]
        "###);
    }
}
