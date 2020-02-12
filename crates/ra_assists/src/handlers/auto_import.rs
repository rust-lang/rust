use crate::{
    assist_ctx::{Assist, AssistCtx},
    insert_use_statement, AssistId,
};
use hir::{
    db::{DefDatabase, HirDatabase},
    AssocContainerId, AssocItem, Crate, Function, ModPath, Module, ModuleDef, PathResolution,
    SourceAnalyzer, Trait, Type,
};
use ra_ide_db::{imports_locator::ImportsLocator, RootDatabase};
use ra_prof::profile;
use ra_syntax::{
    ast::{self, AstNode},
    SyntaxNode,
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
    let auto_import_assets = AutoImportAssets::new(&ctx)?;
    let proposed_imports = auto_import_assets.search_for_imports(ctx.db);
    if proposed_imports.is_empty() {
        return None;
    }

    let assist_group_name = if proposed_imports.len() == 1 {
        format!("Import `{}`", proposed_imports.iter().next().unwrap())
    } else {
        auto_import_assets.get_import_group_message()
    };
    let mut group = ctx.add_assist_group(assist_group_name);
    for import in proposed_imports {
        group.add_assist(AssistId("auto_import"), format!("Import `{}`", &import), |edit| {
            edit.target(auto_import_assets.syntax_under_caret.text_range());
            insert_use_statement(
                &auto_import_assets.syntax_under_caret,
                &auto_import_assets.syntax_under_caret,
                &import,
                edit.text_edit_builder(),
            );
        });
    }
    group.finish()
}

struct AutoImportAssets {
    import_candidate: ImportCandidate,
    module_with_name_to_import: Module,
    syntax_under_caret: SyntaxNode,
}

impl AutoImportAssets {
    fn new(ctx: &AssistCtx) -> Option<Self> {
        if let Some(path_under_caret) = ctx.find_node_at_offset::<ast::Path>() {
            Self::for_regular_path(path_under_caret, &ctx)
        } else {
            Self::for_method_call(ctx.find_node_at_offset()?, &ctx)
        }
    }

    fn for_method_call(method_call: ast::MethodCallExpr, ctx: &AssistCtx) -> Option<Self> {
        let syntax_under_caret = method_call.syntax().to_owned();
        let source_analyzer = ctx.source_analyzer(&syntax_under_caret, None);
        let module_with_name_to_import = source_analyzer.module()?;
        Some(Self {
            import_candidate: ImportCandidate::for_method_call(
                &method_call,
                &source_analyzer,
                ctx.db,
            )?,
            module_with_name_to_import,
            syntax_under_caret,
        })
    }

    fn for_regular_path(path_under_caret: ast::Path, ctx: &AssistCtx) -> Option<Self> {
        let syntax_under_caret = path_under_caret.syntax().to_owned();
        if syntax_under_caret.ancestors().find_map(ast::UseItem::cast).is_some() {
            return None;
        }

        let source_analyzer = ctx.source_analyzer(&syntax_under_caret, None);
        let module_with_name_to_import = source_analyzer.module()?;
        Some(Self {
            import_candidate: ImportCandidate::for_regular_path(
                &path_under_caret,
                &source_analyzer,
                ctx.db,
            )?,
            module_with_name_to_import,
            syntax_under_caret,
        })
    }

    fn get_search_query(&self) -> &str {
        match &self.import_candidate {
            ImportCandidate::UnqualifiedName(name) => name,
            ImportCandidate::QualifierStart(qualifier_start) => qualifier_start,
            ImportCandidate::TraitFunction(_, trait_function_name) => trait_function_name,
            ImportCandidate::TraitMethod(_, trait_method_name) => trait_method_name,
        }
    }

    fn get_import_group_message(&self) -> String {
        match &self.import_candidate {
            ImportCandidate::UnqualifiedName(name) => format!("Import {}", name),
            ImportCandidate::QualifierStart(qualifier_start) => {
                format!("Import {}", qualifier_start)
            }
            ImportCandidate::TraitFunction(_, trait_function_name) => {
                format!("Import a trait for function {}", trait_function_name)
            }
            ImportCandidate::TraitMethod(_, trait_method_name) => {
                format!("Import a trait for method {}", trait_method_name)
            }
        }
    }

    fn search_for_imports(&self, db: &RootDatabase) -> BTreeSet<ModPath> {
        let _p = profile("auto_import::search_for_imports");
        let current_crate = self.module_with_name_to_import.krate();
        ImportsLocator::new(db)
            .find_imports(&self.get_search_query())
            .into_iter()
            .map(|module_def| match &self.import_candidate {
                ImportCandidate::TraitFunction(function_callee, _) => {
                    let mut applicable_traits = Vec::new();
                    if let ModuleDef::Function(located_function) = module_def {
                        let trait_candidates: FxHashSet<_> =
                            Self::get_trait_candidates(db, located_function, current_crate)
                                .into_iter()
                                .map(|trait_candidate| trait_candidate.into())
                                .collect();
                        if !trait_candidates.is_empty() {
                            function_callee.iterate_path_candidates(
                                db,
                                current_crate,
                                &trait_candidates,
                                None,
                                |_, assoc| {
                                    if let AssocContainerId::TraitId(trait_id) = assoc.container(db)
                                    {
                                        applicable_traits.push(
                                            self.module_with_name_to_import.find_use_path(
                                                db,
                                                ModuleDef::Trait(trait_id.into()),
                                            ),
                                        );
                                    };
                                    None::<()>
                                },
                            );
                        };
                    }
                    applicable_traits
                }
                ImportCandidate::TraitMethod(function_callee, _) => {
                    let mut applicable_traits = Vec::new();
                    if let ModuleDef::Function(located_function) = module_def {
                        let trait_candidates: FxHashSet<_> =
                            Self::get_trait_candidates(db, located_function, current_crate)
                                .into_iter()
                                .map(|trait_candidate| trait_candidate.into())
                                .collect();
                        if !trait_candidates.is_empty() {
                            function_callee.iterate_method_candidates(
                                db,
                                current_crate,
                                &trait_candidates,
                                None,
                                |_, funciton| {
                                    if let AssocContainerId::TraitId(trait_id) =
                                        funciton.container(db)
                                    {
                                        applicable_traits.push(
                                            self.module_with_name_to_import.find_use_path(
                                                db,
                                                ModuleDef::Trait(trait_id.into()),
                                            ),
                                        );
                                    };
                                    None::<()>
                                },
                            );
                        }
                    }
                    applicable_traits
                }
                _ => vec![self.module_with_name_to_import.find_use_path(db, module_def)],
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
        let _p = profile("auto_import::get_trait_candidates");
        root_crate
            .dependencies(db)
            .into_iter()
            .map(|dependency| db.crate_def_map(dependency.krate.into()))
            .chain(std::iter::once(db.crate_def_map(root_crate.into())))
            .map(|crate_def_map| {
                crate_def_map
                    .modules
                    .iter()
                    .map(|(_, module_data)| module_data.scope.declarations())
                    .flatten()
                    .filter_map(|module_def_id| match module_def_id.into() {
                        ModuleDef::Trait(trait_candidate)
                            if trait_candidate
                                .items(db)
                                .into_iter()
                                .any(|item| item == AssocItem::Function(called_function)) =>
                        {
                            Some(trait_candidate)
                        }
                        _ => None,
                    })
                    .collect::<FxHashSet<_>>()
            })
            .flatten()
            .collect()
    }
}

#[derive(Debug)]
enum ImportCandidate {
    /// Simple name like 'HashMap'
    UnqualifiedName(String),
    /// First part of the qualified name.
    /// For 'std::collections::HashMap', that will be 'std'.
    QualifierStart(String),
    /// A trait function that has no self parameter.
    /// For 'test_mod::TestEnum::test_function', `Type` is the `test_mod::TestEnum` expression type
    /// and `String`is the `test_function`
    TraitFunction(Type, String),
    /// A trait method with self parameter.
    /// For 'test_enum.test_method()', `Type` is the `test_enum` expression type
    /// and `String` is the `test_method`
    TraitMethod(Type, String),
}

impl ImportCandidate {
    fn for_method_call(
        method_call: &ast::MethodCallExpr,
        source_analyzer: &SourceAnalyzer,
        db: &impl HirDatabase,
    ) -> Option<Self> {
        if source_analyzer.resolve_method_call(method_call).is_some() {
            return None;
        }
        Some(Self::TraitMethod(
            source_analyzer.type_of(db, &method_call.expr()?)?,
            method_call.name_ref()?.syntax().to_string(),
        ))
    }

    fn for_regular_path(
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
                let qualifier_resolution = if qualifier_start_path == qualifier {
                    qualifier_start_resolution
                } else {
                    source_analyzer.resolve_path(db, &qualifier)?
                };
                if let PathResolution::Def(ModuleDef::Adt(function_callee)) = qualifier_resolution {
                    Some(ImportCandidate::TraitFunction(
                        function_callee.ty(db),
                        segment.syntax().to_string(),
                    ))
                } else {
                    None
                }
            } else {
                Some(ImportCandidate::QualifierStart(qualifier_start.syntax().to_string()))
            }
        } else {
            Some(ImportCandidate::UnqualifiedName(
                segment.syntax().descendants().find_map(ast::NameRef::cast)?.syntax().to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{check_assist, check_assist_not_applicable, check_assist_target};

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
    fn not_applicable_for_imported_trait_for_function() {
        check_assist_not_applicable(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub trait TestTrait2 {
                    fn test_function();
                }
                pub enum TestEnum {
                    One,
                    Two,
                }
                impl TestTrait2 for TestEnum {
                    fn test_function() {}
                }
                impl TestTrait for TestEnum {
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
                test_struct.test_meth<|>od()
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
                test_struct.test_meth<|>od()
            }
            ",
        );
    }

    #[test]
    fn not_applicable_for_imported_trait_for_method() {
        check_assist_not_applicable(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    fn test_method(&self);
                }
                pub trait TestTrait2 {
                    fn test_method(&self);
                }
                pub enum TestEnum {
                    One,
                    Two,
                }
                impl TestTrait2 for TestEnum {
                    fn test_method(&self) {}
                }
                impl TestTrait for TestEnum {
                    fn test_method(&self) {}
                }
            }

            use test_mod::TestTrait2;
            fn main() {
                let one = test_mod::TestEnum::One;
                one.test<|>_method();
            }
            ",
        )
    }
}
