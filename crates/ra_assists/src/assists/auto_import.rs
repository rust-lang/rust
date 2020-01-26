use hir::{db::HirDatabase, AsName};
use ra_syntax::{
    ast::{self, AstNode},
    SmolStr, SyntaxElement,
    SyntaxKind::{NAME_REF, USE_ITEM},
    SyntaxNode,
};

use crate::{
    assist_ctx::{ActionBuilder, Assist, AssistCtx},
    auto_import_text_edit, AssistId, ImportsLocator,
};

// Assist: auto_import
//
// If the name is unresolved, provides all possible imports for it.
//
// ```
// fn main() {
//     let map = HashMap<|>::new();
// }
// ```
// ->
// ```
// use std::collections::HashMap;
//
// fn main() {
//     let map = HashMap<|>::new();
// }
// ```
pub(crate) fn auto_import<F: ImportsLocator>(
    ctx: AssistCtx<impl HirDatabase>,
    imports_locator: &mut F,
) -> Option<Assist> {
    let path: ast::Path = ctx.find_node_at_offset()?;
    let module = path.syntax().ancestors().find_map(ast::Module::cast);
    let position = match module.and_then(|it| it.item_list()) {
        Some(item_list) => item_list.syntax().clone(),
        None => {
            let current_file = path.syntax().ancestors().find_map(ast::SourceFile::cast)?;
            current_file.syntax().clone()
        }
    };
    let source_analyzer = ctx.source_analyzer(&position, None);
    let module_with_name_to_import = source_analyzer.module()?;
    let path_to_import = ctx.covering_element().ancestors().find_map(ast::Path::cast)?;
    if source_analyzer.resolve_path(ctx.db, &path_to_import).is_some() {
        return None;
    }

    let name_to_import = &find_applicable_name_ref(ctx.covering_element())?.as_name();
    let proposed_imports = imports_locator
        .find_imports(&name_to_import.to_string())
        .into_iter()
        .filter_map(|module_def| module_with_name_to_import.find_use_path(ctx.db, module_def))
        .filter(|use_path| !use_path.segments.is_empty())
        .take(20)
        .collect::<std::collections::HashSet<_>>();
    if proposed_imports.is_empty() {
        return None;
    }

    ctx.add_assist_group(AssistId("auto_import"), "auto import", || {
        proposed_imports
            .into_iter()
            .map(|import| import_to_action(import.to_string(), &position, &path_to_import))
            .collect()
    })
}

fn find_applicable_name_ref(element: SyntaxElement) -> Option<ast::NameRef> {
    if element.ancestors().find(|ancestor| ancestor.kind() == USE_ITEM).is_some() {
        None
    } else if element.kind() == NAME_REF {
        Some(element.as_node().cloned().and_then(ast::NameRef::cast)?)
    } else {
        let parent = element.parent()?;
        if parent.kind() == NAME_REF {
            Some(ast::NameRef::cast(parent)?)
        } else {
            None
        }
    }
}

fn import_to_action(import: String, position: &SyntaxNode, path: &ast::Path) -> ActionBuilder {
    let mut action_builder = ActionBuilder::default();
    action_builder.label(format!("Import `{}`", &import));
    auto_import_text_edit(
        position,
        &path.syntax().clone(),
        &[SmolStr::new(import)],
        action_builder.text_edit_builder(),
    );
    action_builder
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{
        check_assist_with_imports_locator, check_assist_with_imports_locator_not_applicable,
        TestImportsLocator,
    };

    #[test]
    fn applicable_when_found_an_import() {
        check_assist_with_imports_locator(
            auto_import,
            TestImportsLocator::new,
            r"
            PubStruct<|>

            pub mod PubMod {
                pub struct PubStruct;
            }
            ",
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
    fn applicable_when_found_multiple_imports() {
        check_assist_with_imports_locator(
            auto_import,
            TestImportsLocator::new,
            r"
            PubStruct<|>

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

            PubStruct<|>

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
        check_assist_with_imports_locator_not_applicable(
            auto_import,
            TestImportsLocator::new,
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
        check_assist_with_imports_locator_not_applicable(
            auto_import,
            TestImportsLocator::new,
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
        check_assist_with_imports_locator_not_applicable(
            auto_import,
            TestImportsLocator::new,
            "
            PubStruct<|>",
        );
    }

    #[test]
    fn not_applicable_in_import_statements() {
        check_assist_with_imports_locator_not_applicable(
            auto_import,
            TestImportsLocator::new,
            r"
            use PubStruct<|>;

            pub mod PubMod {
                pub struct PubStruct;
            }",
        );
    }
}
