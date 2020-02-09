use crate::completion::{
    CompletionContext, CompletionItem, CompletionItemKind, CompletionKind, Completions,
};

use hir::{self, Docs};

use ra_assists::utils::get_missing_impl_items;

pub(crate) fn complete_trait_impl(acc: &mut Completions, ctx: &CompletionContext) {
    let impl_block = ctx.impl_block.as_ref();
    let item_list = impl_block.and_then(|i| i.item_list());

    if item_list.is_none() || impl_block.is_none() || ctx.function_syntax.is_some() {
        return;
    }

    let impl_block = impl_block.unwrap();

    for item in get_missing_impl_items(ctx.db, &ctx.analyzer, impl_block) {
        match item {
            hir::AssocItem::Function(f) => add_function_impl(acc, ctx, &f),
            hir::AssocItem::TypeAlias(t) => add_type_alias_impl(acc, ctx, &t),
            _ => {}
        }
    }
}

fn add_function_impl(acc: &mut Completions, ctx: &CompletionContext, func: &hir::Function) {
    use crate::display::FunctionSignature;

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

    let snippet = {
        let mut s = format!("{}", display);
        s.push_str(" {}");
        s
    };

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
}
