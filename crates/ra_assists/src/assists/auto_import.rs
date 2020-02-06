use hir::ModPath;
use ra_syntax::{
    ast::{self, AstNode},
    SyntaxNode,
};

use crate::{
    assist_ctx::{ActionBuilder, Assist, AssistCtx},
    auto_import_text_edit, AssistId,
};
use ra_ide_db::imports_locator::ImportsLocator;

// Assist: auto_import
//
// If the name is unresolved, provides all possible imports for it.
//
// ```
// fn main() {
//     let map = HashMap<|>::new();
// }
// # pub mod std { pub mod collections { pub struct HashMap { } } }
// ```
// ->
// ```
// use std::collections::HashMap;
//
// fn main() {
//     let map = HashMap::new();
// }
// # pub mod std { pub mod collections { pub struct HashMap { } } }
// ```
pub(crate) fn auto_import(ctx: AssistCtx) -> Option<Assist> {
    let path_to_import: ast::Path = ctx.find_node_at_offset()?;
    let path_to_import_syntax = path_to_import.syntax();
    if path_to_import_syntax.ancestors().find_map(ast::UseItem::cast).is_some() {
        return None;
    }
    let name_to_import =
        path_to_import_syntax.descendants().find_map(ast::NameRef::cast)?.syntax().to_string();

    let module = path_to_import_syntax.ancestors().find_map(ast::Module::cast);
    let position = match module.and_then(|it| it.item_list()) {
        Some(item_list) => item_list.syntax().clone(),
        None => {
            let current_file = path_to_import_syntax.ancestors().find_map(ast::SourceFile::cast)?;
            current_file.syntax().clone()
        }
    };
    let source_analyzer = ctx.source_analyzer(&position, None);
    let module_with_name_to_import = source_analyzer.module()?;
    if source_analyzer.resolve_path(ctx.db, &path_to_import).is_some() {
        return None;
    }

    let mut imports_locator = ImportsLocator::new(ctx.db);

    let proposed_imports = imports_locator
        .find_imports(&name_to_import)
        .into_iter()
        .filter_map(|module_def| module_with_name_to_import.find_use_path(ctx.db, module_def))
        .filter(|use_path| !use_path.segments.is_empty())
        .take(20)
        .collect::<std::collections::BTreeSet<_>>();
    if proposed_imports.is_empty() {
        return None;
    }

    ctx.add_assist_group(AssistId("auto_import"), format!("Import {}", name_to_import), || {
        proposed_imports
            .into_iter()
            .map(|import| import_to_action(import, &position, &path_to_import_syntax))
            .collect()
    })
}

fn import_to_action(import: ModPath, position: &SyntaxNode, anchor: &SyntaxNode) -> ActionBuilder {
    let mut action_builder = ActionBuilder::default();
    action_builder.label(format!("Import `{}`", &import));
    auto_import_text_edit(position, anchor, &import, action_builder.text_edit_builder());
    action_builder
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{check_assist, check_assist_not_applicable};

    #[test]
    fn applicable_when_found_an_import() {
        check_assist(
            auto_import,
            r"
            <|>PubStruct

            pub mod PubMod {
                pub struct PubStruct;
            }
            ",
            r"
            <|>use PubMod::PubStruct;

            PubStruct

            pub mod PubMod {
                pub struct PubStruct;
            }
            ",
        );
    }

    #[test]
    fn auto_imports_are_merged() {
        check_assist(
            auto_import,
            r"
            use PubMod::PubStruct1;

            struct Test {
                test: Pub<|>Struct2<u8>,
            }

            pub mod PubMod {
                pub struct PubStruct1;
                pub struct PubStruct2<T> {
                    _t: T,
                }
            }
            ",
            r"
            use PubMod::{PubStruct2, PubStruct1};

            struct Test {
                test: Pub<|>Struct2<u8>,
            }

            pub mod PubMod {
                pub struct PubStruct1;
                pub struct PubStruct2<T> {
                    _t: T,
                }
            }
            ",
        );
    }

    #[test]
    fn applicable_when_found_multiple_imports() {
        check_assist(
            auto_import,
            r"
            PubSt<|>ruct

            pub mod PubMod1 {
                pub struct PubStruct;
            }
            pub mod PubMod2 {
                pub struct PubStruct;
            }
            pub mod PubMod3 {
                pub struct PubStruct;
            }
            ",
            r"
            use PubMod1::PubStruct;

            PubSt<|>ruct

            pub mod PubMod1 {
                pub struct PubStruct;
            }
            pub mod PubMod2 {
                pub struct PubStruct;
            }
            pub mod PubMod3 {
                pub struct PubStruct;
            }
            ",
        );
    }

    #[test]
    fn not_applicable_for_already_imported_types() {
        check_assist_not_applicable(
            auto_import,
            r"
            use PubMod::PubStruct;

            PubStruct<|>

            pub mod PubMod {
                pub struct PubStruct;
            }
            ",
        );
    }

    #[test]
    fn not_applicable_for_types_with_private_paths() {
        check_assist_not_applicable(
            auto_import,
            r"
            PrivateStruct<|>

            pub mod PubMod {
                struct PrivateStruct;
            }
            ",
        );
    }

    #[test]
    fn not_applicable_when_no_imports_found() {
        check_assist_not_applicable(
            auto_import,
            "
            PubStruct<|>",
        );
    }

    #[test]
    fn not_applicable_in_import_statements() {
        check_assist_not_applicable(
            auto_import,
            r"
            use PubStruct<|>;

            pub mod PubMod {
                pub struct PubStruct;
            }",
        );
    }

    #[test]
    fn function_import() {
        check_assist(
            auto_import,
            r"
            test_function<|>

            pub mod PubMod {
                pub fn test_function() {};
            }
            ",
            r"
            use PubMod::test_function;

            test_function<|>

            pub mod PubMod {
                pub fn test_function() {};
            }
            ",
        );
    }
}
