use std::collections::BTreeSet;

use either::Either;
use hir::{
    AsAssocItem, AssocItemContainer, ModPath, Module, ModuleDef, PathResolution, Semantics, Trait,
    Type,
};
use ra_ide_db::{imports_locator, RootDatabase};
use ra_prof::profile;
use ra_syntax::{
    ast::{self, AstNode},
    SyntaxNode,
};
use rustc_hash::FxHashSet;

use crate::{
    utils::insert_use_statement, AssistContext, AssistId, AssistKind, Assists, GroupLabel,
};

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
pub(crate) fn auto_import(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let auto_import_assets = AutoImportAssets::new(ctx)?;
    let proposed_imports = auto_import_assets.search_for_imports(ctx);
    if proposed_imports.is_empty() {
        return None;
    }

    let range = ctx.sema.original_range(&auto_import_assets.syntax_under_caret).range;
    let group = auto_import_assets.get_import_group_message();
    for import in proposed_imports {
        acc.add_group(
            &group,
            AssistId("auto_import", AssistKind::QuickFix),
            format!("Import `{}`", &import),
            range,
            |builder| {
                insert_use_statement(
                    &auto_import_assets.syntax_under_caret,
                    &import,
                    ctx,
                    builder.text_edit_builder(),
                );
            },
        );
    }
    Some(())
}

#[derive(Debug)]
struct AutoImportAssets {
    import_candidate: ImportCandidate,
    module_with_name_to_import: Module,
    syntax_under_caret: SyntaxNode,
}

impl AutoImportAssets {
    fn new(ctx: &AssistContext) -> Option<Self> {
        if let Some(path_under_caret) = ctx.find_node_at_offset_with_descend::<ast::Path>() {
            Self::for_regular_path(path_under_caret, &ctx)
        } else {
            Self::for_method_call(ctx.find_node_at_offset_with_descend()?, &ctx)
        }
    }

    fn for_method_call(method_call: ast::MethodCallExpr, ctx: &AssistContext) -> Option<Self> {
        let syntax_under_caret = method_call.syntax().to_owned();
        let module_with_name_to_import = ctx.sema.scope(&syntax_under_caret).module()?;
        Some(Self {
            import_candidate: ImportCandidate::for_method_call(&ctx.sema, &method_call)?,
            module_with_name_to_import,
            syntax_under_caret,
        })
    }

    fn for_regular_path(path_under_caret: ast::Path, ctx: &AssistContext) -> Option<Self> {
        let syntax_under_caret = path_under_caret.syntax().to_owned();
        if syntax_under_caret.ancestors().find_map(ast::UseItem::cast).is_some() {
            return None;
        }

        let module_with_name_to_import = ctx.sema.scope(&syntax_under_caret).module()?;
        Some(Self {
            import_candidate: ImportCandidate::for_regular_path(&ctx.sema, &path_under_caret)?,
            module_with_name_to_import,
            syntax_under_caret,
        })
    }

    fn get_search_query(&self) -> &str {
        match &self.import_candidate {
            ImportCandidate::UnqualifiedName(name) => name,
            ImportCandidate::QualifierStart(qualifier_start) => qualifier_start,
            ImportCandidate::TraitAssocItem(_, trait_assoc_item_name) => trait_assoc_item_name,
            ImportCandidate::TraitMethod(_, trait_method_name) => trait_method_name,
        }
    }

    fn get_import_group_message(&self) -> GroupLabel {
        let name = match &self.import_candidate {
            ImportCandidate::UnqualifiedName(name) => format!("Import {}", name),
            ImportCandidate::QualifierStart(qualifier_start) => {
                format!("Import {}", qualifier_start)
            }
            ImportCandidate::TraitAssocItem(_, trait_assoc_item_name) => {
                format!("Import a trait for item {}", trait_assoc_item_name)
            }
            ImportCandidate::TraitMethod(_, trait_method_name) => {
                format!("Import a trait for method {}", trait_method_name)
            }
        };
        GroupLabel(name)
    }

    fn search_for_imports(&self, ctx: &AssistContext) -> BTreeSet<ModPath> {
        let _p = profile("auto_import::search_for_imports");
        let db = ctx.db();
        let current_crate = self.module_with_name_to_import.krate();
        imports_locator::find_imports(&ctx.sema, current_crate, &self.get_search_query())
            .into_iter()
            .filter_map(|candidate| match &self.import_candidate {
                ImportCandidate::TraitAssocItem(assoc_item_type, _) => {
                    let located_assoc_item = match candidate {
                        Either::Left(ModuleDef::Function(located_function)) => located_function
                            .as_assoc_item(db)
                            .map(|assoc| assoc.container(db))
                            .and_then(Self::assoc_to_trait),
                        Either::Left(ModuleDef::Const(located_const)) => located_const
                            .as_assoc_item(db)
                            .map(|assoc| assoc.container(db))
                            .and_then(Self::assoc_to_trait),
                        _ => None,
                    }?;

                    let mut trait_candidates = FxHashSet::default();
                    trait_candidates.insert(located_assoc_item.into());

                    assoc_item_type
                        .iterate_path_candidates(
                            db,
                            current_crate,
                            &trait_candidates,
                            None,
                            |_, assoc| Self::assoc_to_trait(assoc.container(db)),
                        )
                        .map(ModuleDef::from)
                        .map(Either::Left)
                }
                ImportCandidate::TraitMethod(function_callee, _) => {
                    let located_assoc_item =
                        if let Either::Left(ModuleDef::Function(located_function)) = candidate {
                            located_function
                                .as_assoc_item(db)
                                .map(|assoc| assoc.container(db))
                                .and_then(Self::assoc_to_trait)
                        } else {
                            None
                        }?;

                    let mut trait_candidates = FxHashSet::default();
                    trait_candidates.insert(located_assoc_item.into());

                    function_callee
                        .iterate_method_candidates(
                            db,
                            current_crate,
                            &trait_candidates,
                            None,
                            |_, function| {
                                Self::assoc_to_trait(function.as_assoc_item(db)?.container(db))
                            },
                        )
                        .map(ModuleDef::from)
                        .map(Either::Left)
                }
                _ => Some(candidate),
            })
            .filter_map(|candidate| match candidate {
                Either::Left(module_def) => {
                    self.module_with_name_to_import.find_use_path(db, module_def)
                }
                Either::Right(macro_def) => {
                    self.module_with_name_to_import.find_use_path(db, macro_def)
                }
            })
            .filter(|use_path| !use_path.segments.is_empty())
            .take(20)
            .collect::<BTreeSet<_>>()
    }

    fn assoc_to_trait(assoc: AssocItemContainer) -> Option<Trait> {
        if let AssocItemContainer::Trait(extracted_trait) = assoc {
            Some(extracted_trait)
        } else {
            None
        }
    }
}

#[derive(Debug)]
enum ImportCandidate {
    /// Simple name like 'HashMap'
    UnqualifiedName(String),
    /// First part of the qualified name.
    /// For 'std::collections::HashMap', that will be 'std'.
    QualifierStart(String),
    /// A trait associated function (with no self parameter) or associated constant.
    /// For 'test_mod::TestEnum::test_function', `Type` is the `test_mod::TestEnum` expression type
    /// and `String` is the `test_function`
    TraitAssocItem(Type, String),
    /// A trait method with self parameter.
    /// For 'test_enum.test_method()', `Type` is the `test_enum` expression type
    /// and `String` is the `test_method`
    TraitMethod(Type, String),
}

impl ImportCandidate {
    fn for_method_call(
        sema: &Semantics<RootDatabase>,
        method_call: &ast::MethodCallExpr,
    ) -> Option<Self> {
        if sema.resolve_method_call(method_call).is_some() {
            return None;
        }
        Some(Self::TraitMethod(
            sema.type_of_expr(&method_call.expr()?)?,
            method_call.name_ref()?.syntax().to_string(),
        ))
    }

    fn for_regular_path(
        sema: &Semantics<RootDatabase>,
        path_under_caret: &ast::Path,
    ) -> Option<Self> {
        if sema.resolve_path(path_under_caret).is_some() {
            return None;
        }

        let segment = path_under_caret.segment()?;
        if let Some(qualifier) = path_under_caret.qualifier() {
            let qualifier_start = qualifier.syntax().descendants().find_map(ast::NameRef::cast)?;
            let qualifier_start_path =
                qualifier_start.syntax().ancestors().find_map(ast::Path::cast)?;
            if let Some(qualifier_start_resolution) = sema.resolve_path(&qualifier_start_path) {
                let qualifier_resolution = if qualifier_start_path == qualifier {
                    qualifier_start_resolution
                } else {
                    sema.resolve_path(&qualifier)?
                };
                if let PathResolution::Def(ModuleDef::Adt(assoc_item_path)) = qualifier_resolution {
                    Some(ImportCandidate::TraitAssocItem(
                        assoc_item_path.ty(sema.db),
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
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

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
            use PubMod::PubStruct;

            PubStruct

            pub mod PubMod {
                pub struct PubStruct;
            }
            ",
        );
    }

    #[test]
    fn applicable_when_found_an_import_in_macros() {
        check_assist(
            auto_import,
            r"
            macro_rules! foo {
                ($i:ident) => { fn foo(a: $i) {} }
            }
            foo!(Pub<|>Struct);

            pub mod PubMod {
                pub struct PubStruct;
            }
            ",
            r"
            use PubMod::PubStruct;

            macro_rules! foo {
                ($i:ident) => { fn foo(a: $i) {} }
            }
            foo!(PubStruct);

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
                test: PubStruct2<u8>,
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
            use PubMod3::PubStruct;

            PubStruct

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

            test_function

            pub mod PubMod {
                pub fn test_function() {};
            }
            ",
        );
    }

    #[test]
    fn macro_import() {
        check_assist(
            auto_import,
            r"
//- /lib.rs crate:crate_with_macro
#[macro_export]
macro_rules! foo {
    () => ()
}

//- /main.rs crate:main deps:crate_with_macro
fn main() {
    foo<|>
}
",
            r"use crate_with_macro::foo;

fn main() {
    foo
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
                TestStruct::test_function
            }
            ",
        );
    }

    #[test]
    fn associated_struct_const() {
        check_assist(
            auto_import,
            r"
            mod test_mod {
                pub struct TestStruct {}
                impl TestStruct {
                    const TEST_CONST: u8 = 42;
                }
            }

            fn main() {
                TestStruct::TEST_CONST<|>
            }
            ",
            r"
            use test_mod::TestStruct;

            mod test_mod {
                pub struct TestStruct {}
                impl TestStruct {
                    const TEST_CONST: u8 = 42;
                }
            }

            fn main() {
                TestStruct::TEST_CONST
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
                test_mod::TestStruct::test_function
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
    fn associated_trait_const() {
        check_assist(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    const TEST_CONST: u8;
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    const TEST_CONST: u8 = 42;
                }
            }

            fn main() {
                test_mod::TestStruct::TEST_CONST<|>
            }
            ",
            r"
            use test_mod::TestTrait;

            mod test_mod {
                pub trait TestTrait {
                    const TEST_CONST: u8;
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    const TEST_CONST: u8 = 42;
                }
            }

            fn main() {
                test_mod::TestStruct::TEST_CONST
            }
            ",
        );
    }

    #[test]
    fn not_applicable_for_imported_trait_for_const() {
        check_assist_not_applicable(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    const TEST_CONST: u8;
                }
                pub trait TestTrait2 {
                    const TEST_CONST: f64;
                }
                pub enum TestEnum {
                    One,
                    Two,
                }
                impl TestTrait2 for TestEnum {
                    const TEST_CONST: f64 = 42.0;
                }
                impl TestTrait for TestEnum {
                    const TEST_CONST: u8 = 42;
                }
            }

            use test_mod::TestTrait2;
            fn main() {
                test_mod::TestEnum::TEST_CONST<|>;
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
                test_struct.test_method()
            }
            ",
        );
    }

    #[test]
    fn trait_method_cross_crate() {
        check_assist(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                let test_struct = dep::test_mod::TestStruct {};
                test_struct.test_meth<|>od()
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    fn test_method(&self);
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_method(&self) {}
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait;

            fn main() {
                let test_struct = dep::test_mod::TestStruct {};
                test_struct.test_method()
            }
            ",
        );
    }

    #[test]
    fn assoc_fn_cross_crate() {
        check_assist(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                dep::test_mod::TestStruct::test_func<|>tion
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait;

            fn main() {
                dep::test_mod::TestStruct::test_function
            }
            ",
        );
    }

    #[test]
    fn assoc_const_cross_crate() {
        check_assist(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                dep::test_mod::TestStruct::CONST<|>
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    const CONST: bool;
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    const CONST: bool = true;
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait;

            fn main() {
                dep::test_mod::TestStruct::CONST
            }
            ",
        );
    }

    #[test]
    fn assoc_fn_as_method_cross_crate() {
        check_assist_not_applicable(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                let test_struct = dep::test_mod::TestStruct {};
                test_struct.test_func<|>tion()
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }
            ",
        );
    }

    #[test]
    fn private_trait_cross_crate() {
        check_assist_not_applicable(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                let test_struct = dep::test_mod::TestStruct {};
                test_struct.test_meth<|>od()
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                trait TestTrait {
                    fn test_method(&self);
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_method(&self) {}
                }
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

    #[test]
    fn dep_import() {
        check_assist(
            auto_import,
            r"
//- /lib.rs crate:dep
pub struct Struct;

//- /main.rs crate:main deps:dep
fn main() {
    Struct<|>
}
",
            r"use dep::Struct;

fn main() {
    Struct
}
",
        );
    }

    #[test]
    fn whole_segment() {
        // Tests that only imports whose last segment matches the identifier get suggested.
        check_assist(
            auto_import,
            r"
//- /lib.rs crate:dep
pub mod fmt {
    pub trait Display {}
}

pub fn panic_fmt() {}

//- /main.rs crate:main deps:dep
struct S;

impl f<|>mt::Display for S {}
",
            r"use dep::fmt;

struct S;

impl fmt::Display for S {}
",
        );
    }

    #[test]
    fn macro_generated() {
        // Tests that macro-generated items are suggested from external crates.
        check_assist(
            auto_import,
            r"
//- /lib.rs crate:dep
macro_rules! mac {
    () => {
        pub struct Cheese;
    };
}

mac!();

//- /main.rs crate:main deps:dep
fn main() {
    Cheese<|>;
}
",
            r"use dep::Cheese;

fn main() {
    Cheese;
}
",
        );
    }

    #[test]
    fn casing() {
        // Tests that differently cased names don't interfere and we only suggest the matching one.
        check_assist(
            auto_import,
            r"
//- /lib.rs crate:dep
pub struct FMT;
pub struct fmt;

//- /main.rs crate:main deps:dep
fn main() {
    FMT<|>;
}
",
            r"use dep::FMT;

fn main() {
    FMT;
}
",
        );
    }
}
