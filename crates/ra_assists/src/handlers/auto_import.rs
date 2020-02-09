use ra_ide_db::imports_locator::ImportsLocator;
use ra_syntax::ast::{self, AstNode};

use crate::{
    assist_ctx::{Assist, AssistCtx},
    insert_use_statement, AssistId,
};
use std::collections::BTreeSet;

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
    let path_under_caret: ast::Path = ctx.find_node_at_offset()?;
    if path_under_caret.syntax().ancestors().find_map(ast::UseItem::cast).is_some() {
        return None;
    }

    let module = path_under_caret.syntax().ancestors().find_map(ast::Module::cast);
    let position = match module.and_then(|it| it.item_list()) {
        Some(item_list) => item_list.syntax().clone(),
        None => {
            let current_file =
                path_under_caret.syntax().ancestors().find_map(ast::SourceFile::cast)?;
            current_file.syntax().clone()
        }
    };
    let source_analyzer = ctx.source_analyzer(&position, None);
    let module_with_name_to_import = source_analyzer.module()?;

    let name_ref_to_import =
        path_under_caret.syntax().descendants().find_map(ast::NameRef::cast)?;
    if dbg!(source_analyzer
        .resolve_path(ctx.db, &name_ref_to_import.syntax().ancestors().find_map(ast::Path::cast)?))
    .is_some()
    {
        return None;
    }

    let name_to_import = name_ref_to_import.syntax().to_string();
    let proposed_imports = ImportsLocator::new(ctx.db)
        .find_imports(&name_to_import)
        .into_iter()
        .filter_map(|module_def| module_with_name_to_import.find_use_path(ctx.db, module_def))
        .filter(|use_path| !use_path.segments.is_empty())
        .take(20)
        .collect::<BTreeSet<_>>();

    if proposed_imports.is_empty() {
        return None;
    }

    let mut group = ctx.add_assist_group(format!("Import {}", name_to_import));
    for import in proposed_imports {
        group.add_assist(AssistId("auto_import"), format!("Import `{}`", &import), |edit| {
            edit.target(path_under_caret.syntax().text_range());
            insert_use_statement(
                &position,
                path_under_caret.syntax(),
                &import,
                edit.text_edit_builder(),
            );
        });
    }
    group.finish()
}

#[cfg(test)]
mod tests {
    use crate::helpers::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

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

    #[test]
    fn auto_import_target() {
        check_assist_target(
            auto_import,
            r"
            struct AssistInfo {
                group_label: Option<<|>GroupLabel>,
            }

            mod m { pub struct GroupLabel; }
            ",
            "GroupLabel",
        )
    }

    #[test]
    fn not_applicable_when_path_start_is_imported() {
        check_assist_not_applicable(
            auto_import,
            r"
            pub mod mod1 {
                pub mod mod2 {
                    pub mod mod3 {
                        pub struct TestStruct;
                    }
                }
            }

            use mod1::mod2;
            fn main() {
                mod2::mod3::TestStruct<|>
            }
            ",
        );
    }

    #[test]
    fn not_applicable_for_imported_function() {
        check_assist_not_applicable(
            auto_import,
            r"
            pub mod test_mod {
                pub fn test_function() {}
            }

            use test_mod::test_function;
            fn main() {
                test_function<|>
            }
            ",
        );
    }
}
