use crate::completion::{CompletionContext, Completions};

use ast::{ NameOwner };
use hir::{ self, db::HirDatabase };

use ra_syntax::{ ast, ast::AstNode };

pub(crate) fn complete_trait_impl(acc: &mut Completions, ctx: &CompletionContext) {
    let item_list = ast::ItemList::cast(ctx.token.parent());
    let impl_block = item_list
        .clone()
        .and_then(|i| i.syntax().parent())
        .and_then(|p| ast::ImplBlock::cast(p));

    if item_list.is_none() || impl_block.is_none() {
        return;
    }

    let item_list = item_list.unwrap();
    let impl_block = impl_block.unwrap();

    let target_trait = resolve_target_trait(ctx.db, &ctx.analyzer, &impl_block);
    if target_trait.is_none() {
        return;
    }

    let trait_ = target_trait.unwrap();

    let trait_items = trait_.items(ctx.db);
    let missing_items = trait_items
        .iter()
        .filter(|i| {
            match i {
                hir::AssocItem::Function(f) => {
                    let f_name = f.name(ctx.db).to_string();

                    item_list
                        .impl_items()
                        .find(|impl_item| {
                            match impl_item {
                                ast::ImplItem::FnDef(impl_f) => {
                                    if let Some(n) = impl_f.name() { 
                                        f_name == n.syntax().to_string()
                                    } else { 
                                        false
                                    }
                                },
                                _ => false
                            }
                        }).is_none()
                },
                hir::AssocItem::Const(c) => {
                    let c_name = c.name(ctx.db)
                        .map(|f| f.to_string());

                    if c_name.is_none() {
                        return false;
                    }

                    let c_name = c_name.unwrap();

                    item_list
                        .impl_items()
                        .find(|impl_item| {
                            match impl_item {
                                ast::ImplItem::ConstDef(c) => {
                                    if let Some(n) = c.name() { 
                                        c_name == n.syntax().to_string()
                                    } else { 
                                        false
                                    }
                                },
                                _ => false
                            }
                        }).is_none()
                },
                hir::AssocItem::TypeAlias(t) => {
                    let t_name = t.name(ctx.db).to_string();

                    item_list
                        .impl_items()
                        .find(|impl_item| {
                            match impl_item {
                                ast::ImplItem::TypeAliasDef(t) => {
                                    if let Some(n) = t.name() { 
                                        t_name == n.syntax().to_string()
                                    } else { 
                                        false
                                    }
                                },
                                _ => false
                            }
                        }).is_none()
                }
            }
        });

    for item in missing_items {
        match item {
            hir::AssocItem::Function(f) => acc.add_function_impl(ctx, f),
            _ => {}
        }
    }
}

fn resolve_target_trait(
    db: &impl HirDatabase,
    analyzer: &hir::SourceAnalyzer,
    impl_block: &ast::ImplBlock
) -> Option<hir::Trait> {
    let ast_path = impl_block
        .target_trait()
        .map(|it| it.syntax().clone())
        .and_then(ast::PathType::cast)?
        .path()?;

    match analyzer.resolve_path(db, &ast_path) {
        Some(hir::PathResolution::Def(hir::ModuleDef::Trait(def))) => {
            Some(def)
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
                insert: "fn foo() { $0}",
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
                insert: "fn bar() { $0}",
                kind: Function,
                lookup: "bar",
            },
        ]
        "###);
    }
}