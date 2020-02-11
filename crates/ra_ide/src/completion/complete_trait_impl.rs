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
    // it is possible to have a parent `fn` and `impl` block. Ignore completion
    // attempts from within a `fn` block.
    if ctx.function_syntax.is_some() {
        return;
    }

    if let Some(ref impl_block) = ctx.impl_block {
        for item in get_missing_impl_items(ctx.db, &ctx.analyzer, impl_block) {
            match item {
                hir::AssocItem::Function(f) => add_function_impl(acc, ctx, &f),
                hir::AssocItem::TypeAlias(t) => add_type_alias_impl(acc, ctx, &t),
                hir::AssocItem::Const(c) => add_const_impl(acc, ctx, &c),
            }
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
                <|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "fn foo()",
                source_range: [138; 138),
                delete: [138; 138),
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

                <|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "fn bar()",
                source_range: [193; 193),
                delete: [193; 193),
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
                <|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "fn foo()",
                source_range: [141; 141),
                delete: [141; 141),
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
                <|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "fn foo()",
                source_range: [163; 163),
                delete: [163; 163),
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
                <|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "type SomeType = ",
                source_range: [119; 119),
                delete: [119; 119),
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
                <|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "const SOME_CONST: u16 = ",
                source_range: [127; 127),
                delete: [127; 127),
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
                <|>
            }
            ",
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "const SOME_CONST: u16 = ",
                source_range: [132; 132),
                delete: [132; 132),
                insert: "const SOME_CONST: u16 = ",
                kind: Const,
            },
        ]
        "###);
    }
}
