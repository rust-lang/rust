use crate::completion::{CompletionContext, Completions, CompletionItem, CompletionKind, CompletionItemKind};

use ast::{ NameOwner };
use hir::{ self, db::HirDatabase };

use ra_syntax::{ SyntaxKind, ast, ast::AstNode, TextRange };

pub(crate) fn complete_trait_impl(acc: &mut Completions, ctx: &CompletionContext) {
    let impl_block = ctx.impl_block.as_ref();
    let item_list = impl_block.and_then(|i| i.item_list());

    if item_list.is_none() 
    || impl_block.is_none() 
    || ctx.function_syntax.is_some() {
        return;
    }

    let item_list = item_list.unwrap();
    let impl_block = impl_block.unwrap();

    let target_trait = resolve_target_trait(ctx.db, &ctx.analyzer, &impl_block);
    if target_trait.is_none() {
        return;
    }

    // for cases where the user has already started writing the function def, navigate
    // the previous tokens in order to find the location of that token so that we may 
    // replace it with our completion.
    let start_position = {        
        let mut prev_token = ctx.token
            .prev_token()
            .clone();

        while let Some(token) = &prev_token {
            match token.kind() {
                SyntaxKind::FN_KW => break,

                // todo:    attempt to find a better way of determining when to stop as
                //          the following feels sketchy.
                SyntaxKind::IMPL_KW |
                SyntaxKind::L_CURLY |
                SyntaxKind::R_CURLY => {
                    prev_token = None;
                    break;
                }
                _ => {}
            }

            prev_token = token.prev_token().clone();
        }

        prev_token
            .map(|t| t.text_range())
            .unwrap_or(ctx.source_range())
    };

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
            hir::AssocItem::Function(f) => add_function_impl(acc, ctx, f, start_position),
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

pub(crate) fn add_function_impl(acc: &mut Completions, ctx: &CompletionContext, func: &hir::Function, start: TextRange) {
    use crate::display::FunctionSignature;

    let display = FunctionSignature::from_hir(ctx.db, func.clone());

    let func_name = func.name(ctx.db);

    let label = if func.params(ctx.db).len() > 0 {
        format!("fn {}(..)", func_name.to_string())
    } else {
        format!("fn {}()", func_name.to_string())
    };

    let builder = CompletionItem::new(CompletionKind::Magic, start, label.clone())
        .lookup_by(label);

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

    builder
        .insert_text(snippet)
        .kind(completion_kind)
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