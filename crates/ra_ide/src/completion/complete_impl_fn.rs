
use crate::completion::{CompletionContext, Completions};

use hir::{ self, db::HirDatabase, HasSource };

use ra_syntax::{ ast, ast::AstNode };

pub(crate) fn complete_impl_fn(acc: &mut Completions, ctx: &CompletionContext) {
    let impl_trait = ast::ItemList::cast(ctx.token.parent())
        .and_then(|item_list| item_list.syntax().parent())
        .and_then(|item_list_parent| ast::ImplBlock::cast(item_list_parent))
        .and_then(|impl_block| resolve_target_trait(ctx.db, &ctx.analyzer, &impl_block));

    if let Some(x) = &impl_trait {
        for trait_item in x.0.items(ctx.db) {
            match trait_item {
                hir::AssocItem::Function(f) => acc.add_function_impl(ctx, f),
                _ => {}
            }
        }
    }
}

fn resolve_target_trait(
    db: &impl HirDatabase,
    analyzer: &hir::SourceAnalyzer,
    impl_block: &ast::ImplBlock
) -> Option<(hir::Trait, ast::TraitDef)> {
    let ast_path = impl_block
        .target_trait()
        .map(|it| it.syntax().clone())
        .and_then(ast::PathType::cast)?
        .path()?;

    match analyzer.resolve_path(db, &ast_path) {
        Some(hir::PathResolution::Def(hir::ModuleDef::Trait(def))) => {
            Some((def, def.source(db).value))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::completion::{do_completion, CompletionItem, CompletionKind};
    use insta::assert_debug_snapshot;

    fn complete(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Reference)
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
                insert: "fn foo() { $0 }",
                kind: Function,
                lookup: "foo",
            },
        ]
        "###);
    }
}