use ra_ide_db::{imports_locator::ImportsLocator, RootDatabase};
use ra_syntax::ast::{self, AstNode};

use crate::{
    assist_ctx::{Assist, AssistCtx},
    insert_use_statement, AssistId,
};
use ast::{FnDefOwner, ModuleItem, ModuleItemOwner};
use hir::{
    db::{DefDatabase, HirDatabase},
    Adt, AssocContainerId, Crate, Function, HasSource, InFile, ModPath, Module, ModuleDef,
    PathResolution, SourceAnalyzer, SourceBinder, Trait,
};
use rustc_hash::FxHashSet;
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

    let import_candidate = ImportCandidate::new(&path_under_caret, &source_analyzer, ctx.db)?;
    let proposed_imports = import_candidate.search_for_imports(ctx.db, module_with_name_to_import);
    if proposed_imports.is_empty() {
        return None;
    }

    let mut group = ctx.add_assist_group(format!("Import {}", import_candidate.get_search_query()));
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

#[derive(Debug)]
// TODO kb rustdocs
enum ImportCandidate {
    UnqualifiedName(ast::NameRef),
    QualifierStart(ast::NameRef),
    TraitFunction(Adt, ast::PathSegment),
}

impl ImportCandidate {
    // TODO kb refactor this mess
    fn new(
        path_under_caret: &ast::Path,
        source_analyzer: &SourceAnalyzer,
        db: &impl HirDatabase,
    ) -> Option<Self> {
        if source_analyzer.resolve_path(db, path_under_caret).is_some() {
            return None;
        }

        let segment = path_under_caret.segment()?;
        if let Some(qualifier) = path_under_caret.qualifier() {
            let qualifier_start = qualifier.syntax().descendants().find_map(ast::NameRef::cast)?;
            let qualifier_start_path =
                qualifier_start.syntax().ancestors().find_map(ast::Path::cast)?;
            if let Some(qualifier_start_resolution) =
                source_analyzer.resolve_path(db, &qualifier_start_path)
            {
                let qualifier_resolution = if &qualifier_start_path == path_under_caret {
                    qualifier_start_resolution
                } else {
                    source_analyzer.resolve_path(db, &qualifier)?
                };
                if let PathResolution::Def(ModuleDef::Adt(function_callee)) = qualifier_resolution {
                    Some(ImportCandidate::TraitFunction(function_callee, segment))
                } else {
                    None
                }
            } else {
                Some(ImportCandidate::QualifierStart(qualifier_start))
            }
        } else {
            if source_analyzer.resolve_path(db, path_under_caret).is_none() {
                Some(ImportCandidate::UnqualifiedName(
                    segment.syntax().descendants().find_map(ast::NameRef::cast)?,
                ))
            } else {
                None
            }
        }
    }

    fn get_search_query(&self) -> String {
        match self {
            ImportCandidate::UnqualifiedName(name_ref)
            | ImportCandidate::QualifierStart(name_ref) => name_ref.syntax().to_string(),
            ImportCandidate::TraitFunction(_, trait_function) => {
                trait_function.syntax().to_string()
            }
        }
    }

    fn search_for_imports(
        &self,
        db: &RootDatabase,
        module_with_name_to_import: Module,
    ) -> BTreeSet<ModPath> {
        ImportsLocator::new(db)
            .find_imports(&self.get_search_query())
            .into_iter()
            .map(|module_def| match self {
                ImportCandidate::TraitFunction(function_callee, _) => {
                    let mut applicable_traits = Vec::new();
                    if let ModuleDef::Function(located_function) = module_def {
                        let trait_candidates = Self::get_trait_candidates(
                            db,
                            located_function,
                            module_with_name_to_import.krate(),
                        )
                        .into_iter()
                        .map(|trait_candidate| trait_candidate.into())
                        .collect();

                        function_callee.ty(db).iterate_path_candidates(
                            db,
                            module_with_name_to_import.krate(),
                            &trait_candidates,
                            None,
                            |_, assoc| {
                                if let AssocContainerId::TraitId(trait_id) = assoc.container(db) {
                                    applicable_traits.push(
                                        module_with_name_to_import
                                            .find_use_path(db, ModuleDef::Trait(trait_id.into())),
                                    );
                                };
                                None::<()>
                            },
                        );
                    }
                    applicable_traits
                }
                _ => vec![module_with_name_to_import.find_use_path(db, module_def)],
            })
            .flatten()
            .filter_map(std::convert::identity)
            .filter(|use_path| !use_path.segments.is_empty())
            .take(20)
            .collect::<BTreeSet<_>>()
    }

    fn get_trait_candidates(
        db: &RootDatabase,
        called_function: Function,
        root_crate: Crate,
    ) -> FxHashSet<Trait> {
        let mut source_binder = SourceBinder::new(db);
        root_crate
            .dependencies(db)
            .into_iter()
            .map(|dependency| db.crate_def_map(dependency.krate.into()))
            .chain(std::iter::once(db.crate_def_map(root_crate.into())))
            .map(|crate_def_map| {
                crate_def_map
                    .modules
                    .iter()
                    .filter_map(|(_, module_data)| module_data.declaration_source(db))
                    .filter_map(|in_file_module| {
                        Some((in_file_module.file_id, in_file_module.value.item_list()?.items()))
                    })
                    .map(|(file_id, item_list)| {
                        let mut if_file_trait_defs = Vec::new();
                        for module_item in item_list {
                            if let ModuleItem::TraitDef(trait_def) = module_item {
                                if let Some(item_list) = trait_def.item_list() {
                                    if item_list
                                        .functions()
                                        .any(|fn_def| fn_def == called_function.source(db).value)
                                    {
                                        if_file_trait_defs.push(InFile::new(file_id, trait_def))
                                    }
                                }
                            }
                        }
                        if_file_trait_defs
                    })
                    .flatten()
                    .filter_map(|in_file_trait_def| source_binder.to_def(in_file_trait_def))
                    .collect::<FxHashSet<_>>()
            })
            .flatten()
            .collect()
    }
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

    #[test]
    fn associated_struct_function() {
        check_assist(
            auto_import,
            r"
            mod test_mod {
                pub struct TestStruct {}
                impl TestStruct {
                    pub fn test_function() {}
                }
            }

            fn main() {
                TestStruct::test_function<|>
            }
            ",
            r"
            use test_mod::TestStruct;

            mod test_mod {
                pub struct TestStruct {}
                impl TestStruct {
                    pub fn test_function() {}
                }
            }

            fn main() {
                TestStruct::test_function<|>
            }
            ",
        );
    }

    #[test]
    fn associated_trait_function() {
        check_assist(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }

            fn main() {
                test_mod::TestStruct::test_function<|>
            }
            ",
            r"
            use test_mod::TestTrait;

            mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }

            fn main() {
                test_mod::TestStruct::test_function<|>
            }
            ",
        );
    }

    #[test]
    fn not_applicable_for_imported_trait() {
        check_assist_not_applicable(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    fn test_method(&self);
                    fn test_function();
                }

                pub trait TestTrait2 {
                    fn test_method(&self);
                    fn test_function();
                }
                pub enum TestEnum {
                    One,
                    Two,
                }

                impl TestTrait2 for TestEnum {
                    fn test_method(&self) {}
                    fn test_function() {}
                }

                impl TestTrait for TestEnum {
                    fn test_method(&self) {}
                    fn test_function() {}
                }
            }

            use test_mod::TestTrait2;
            fn main() {
                test_mod::TestEnum::test_function<|>;
            }
            ",
        )
    }

    #[test]
    fn trait_method() {
        check_assist(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    fn test_method(&self);
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_method(&self) {}
                }
            }

            fn main() {
                let test_struct = test_mod::TestStruct {};
                test_struct.test_method<|>
            }
            ",
            r"
            use test_mod::TestTrait;

            mod test_mod {
                pub trait TestTrait {
                    fn test_method(&self);
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_method(&self) {}
                }
            }

            fn main() {
                let test_struct = test_mod::TestStruct {};
                test_struct.test_method<|>
            }
            ",
        );
    }
}
