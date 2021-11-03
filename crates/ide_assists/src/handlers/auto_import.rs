use ide_db::helpers::{
    import_assets::{ImportAssets, ImportCandidate},
    insert_use::{insert_use, ImportScope},
    mod_path_to_ast,
};
use syntax::{ast, AstNode, AstToken, NodeOrToken, SyntaxElement};

use crate::{AssistContext, AssistId, AssistKind, Assists, GroupLabel};

// Feature: Auto Import
//
// Using the `auto-import` assist it is possible to insert missing imports for unresolved items.
// When inserting an import it will do so in a structured manner by keeping imports grouped,
// separated by a newline in the following order:
//
// - `std` and `core`
// - External Crates
// - Current Crate, paths prefixed by `crate`
// - Current Module, paths prefixed by `self`
// - Super Module, paths prefixed by `super`
//
// Example:
// ```rust
// use std::fs::File;
//
// use itertools::Itertools;
// use syntax::ast;
//
// use crate::utils::insert_use;
//
// use self::auto_import;
//
// use super::AssistContext;
// ```
//
// .Import Granularity
//
// It is possible to configure how use-trees are merged with the `importGranularity` setting.
// It has the following configurations:
//
// - `crate`: Merge imports from the same crate into a single use statement. This kind of
//  nesting is only supported in Rust versions later than 1.24.
// - `module`: Merge imports from the same module into a single use statement.
// - `item`: Don't merge imports at all, creating one import per item.
// - `preserve`: Do not change the granularity of any imports. For auto-import this has the same
//  effect as `item`.
//
// In `VS Code` the configuration for this is `rust-analyzer.assist.importGranularity`.
//
// .Import Prefix
//
// The style of imports in the same crate is configurable through the `importPrefix` setting.
// It has the following configurations:
//
// - `by_crate`: This setting will force paths to be always absolute, starting with the `crate`
//  prefix, unless the item is defined outside of the current crate.
// - `by_self`: This setting will force paths that are relative to the current module to always
//  start with `self`. This will result in paths that always start with either `crate`, `self`,
//  `super` or an extern crate identifier.
// - `plain`: This setting does not impose any restrictions in imports.
//
// In `VS Code` the configuration for this is `rust-analyzer.assist.importPrefix`.
//
// image::https://user-images.githubusercontent.com/48062697/113020673-b85be580-917a-11eb-9022-59585f35d4f8.gif[]

// Assist: auto_import
//
// If the name is unresolved, provides all possible imports for it.
//
// ```
// fn main() {
//     let map = HashMap$0::new();
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
    let (import_assets, syntax_under_caret) = find_importable_node(ctx)?;
    let mut proposed_imports =
        import_assets.search_for_imports(&ctx.sema, ctx.config.insert_use.prefix_kind);
    if proposed_imports.is_empty() {
        return None;
    }

    let range = match &syntax_under_caret {
        NodeOrToken::Node(node) => ctx.sema.original_range(node).range,
        NodeOrToken::Token(token) => token.text_range(),
    };
    let group_label = group_label(import_assets.import_candidate());
    let scope = ImportScope::find_insert_use_container(
        &match syntax_under_caret {
            NodeOrToken::Node(it) => it,
            NodeOrToken::Token(it) => it.parent()?,
        },
        &ctx.sema,
    )?;

    // we aren't interested in different namespaces
    proposed_imports.dedup_by(|a, b| a.import_path == b.import_path);
    for import in proposed_imports {
        acc.add_group(
            &group_label,
            AssistId("auto_import", AssistKind::QuickFix),
            format!("Import `{}`", import.import_path),
            range,
            |builder| {
                let scope = match scope.clone() {
                    ImportScope::File(it) => ImportScope::File(builder.make_mut(it)),
                    ImportScope::Module(it) => ImportScope::Module(builder.make_mut(it)),
                    ImportScope::Block(it) => ImportScope::Block(builder.make_mut(it)),
                };
                insert_use(&scope, mod_path_to_ast(&import.import_path), &ctx.config.insert_use);
            },
        );
    }
    Some(())
}

pub(super) fn find_importable_node(ctx: &AssistContext) -> Option<(ImportAssets, SyntaxElement)> {
    if let Some(path_under_caret) = ctx.find_node_at_offset_with_descend::<ast::Path>() {
        ImportAssets::for_exact_path(&path_under_caret, &ctx.sema)
            .zip(Some(path_under_caret.syntax().clone().into()))
    } else if let Some(method_under_caret) =
        ctx.find_node_at_offset_with_descend::<ast::MethodCallExpr>()
    {
        ImportAssets::for_method_call(&method_under_caret, &ctx.sema)
            .zip(Some(method_under_caret.syntax().clone().into()))
    } else if let Some(pat) = ctx
        .find_node_at_offset_with_descend::<ast::IdentPat>()
        .filter(ast::IdentPat::is_simple_ident)
    {
        ImportAssets::for_ident_pat(&ctx.sema, &pat).zip(Some(pat.syntax().clone().into()))
    } else {
        // FIXME: Descend?
        let ident = ctx.find_token_at_offset()?;
        ImportAssets::for_derive_ident(&ctx.sema, &ident).zip(Some(ident.syntax().clone().into()))
    }
}

fn group_label(import_candidate: &ImportCandidate) -> GroupLabel {
    let name = match import_candidate {
        ImportCandidate::Path(candidate) => format!("Import {}", candidate.name.text()),
        ImportCandidate::TraitAssocItem(candidate) => {
            format!("Import a trait for item {}", candidate.assoc_item_name.text())
        }
        ImportCandidate::TraitMethod(candidate) => {
            format!("Import a trait for method {}", candidate.assoc_item_name.text())
        }
    };
    GroupLabel(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn not_applicable_if_scope_inside_macro() {
        check_assist_not_applicable(
            auto_import,
            r"
mod bar {
    pub struct Baz;
}
macro_rules! foo {
    ($it:ident) => {
        mod __ {
            fn __(x: $it) {}
        }
    };
}
foo! {
    Baz$0
}
",
        );
    }

    #[test]
    fn applicable_in_attributes() {
        check_assist(
            auto_import,
            r"
//- proc_macros: identity
#[proc_macros::identity]
mod foo {
    mod bar {
        const _: Baz$0 = ();
    }
}
mod baz {
    pub struct Baz;
}
",
            r"
#[proc_macros::identity]
mod foo {
    mod bar {
        use crate::baz::Baz;

        const _: Baz = ();
    }
}
mod baz {
    pub struct Baz;
}
",
        );
    }

    #[test]
    fn applicable_when_found_an_import_partial() {
        check_assist(
            auto_import,
            r"
            mod std {
                pub mod fmt {
                    pub struct Formatter;
                }
            }

            use std::fmt;

            $0Formatter
            ",
            r"
            mod std {
                pub mod fmt {
                    pub struct Formatter;
                }
            }

            use std::fmt::{self, Formatter};

            Formatter
            ",
        );
    }

    #[test]
    fn applicable_when_found_an_import() {
        check_assist(
            auto_import,
            r"
            $0PubStruct

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
            foo!(Pub$0Struct);

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
    fn applicable_when_found_multiple_imports() {
        check_assist(
            auto_import,
            r"
            PubSt$0ruct

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

            PubStruct$0

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
            PrivateStruct$0

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
            PubStruct$0",
        );
    }

    #[test]
    fn not_applicable_in_import_statements() {
        check_assist_not_applicable(
            auto_import,
            r"
            use PubStruct$0;

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
            test_function$0

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
    foo$0
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
                group_label: Option<$0GroupLabel>,
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
                mod2::mod3::TestStruct$0
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
                test_function$0
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
                TestStruct::test_function$0
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
                TestStruct::TEST_CONST$0
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
                test_mod::TestStruct::test_function$0
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
                test_mod::TestEnum::test_function$0;
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
                test_mod::TestStruct::TEST_CONST$0
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
                test_mod::TestEnum::TEST_CONST$0;
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
                test_struct.test_meth$0od()
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
                test_struct.test_meth$0od()
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
                dep::test_mod::TestStruct::test_func$0tion
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
                dep::test_mod::TestStruct::CONST$0
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
                test_struct.test_func$0tion()
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
                test_struct.test_meth$0od()
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
                one.test$0_method();
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
    Struct$0
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

impl f$0mt::Display for S {}
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
    Cheese$0;
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
    FMT$0;
}
",
            r"use dep::FMT;

fn main() {
    FMT;
}
",
        );
    }

    #[test]
    fn inner_items() {
        check_assist(
            auto_import,
            r#"
mod baz {
    pub struct Foo {}
}

mod bar {
    fn bar() {
        Foo$0;
        println!("Hallo");
    }
}
"#,
            r#"
mod baz {
    pub struct Foo {}
}

mod bar {
    use crate::baz::Foo;

    fn bar() {
        Foo;
        println!("Hallo");
    }
}
"#,
        );
    }

    #[test]
    fn uses_abs_path_with_extern_crate_clash() {
        cov_mark::check!(ambiguous_crate_start);
        check_assist(
            auto_import,
            r#"
//- /main.rs crate:main deps:foo
mod foo {}

const _: () = {
    Foo$0
};
//- /foo.rs crate:foo
pub struct Foo
"#,
            r#"
use ::foo::Foo;

mod foo {}

const _: () = {
    Foo
};
"#,
        );
    }

    #[test]
    fn works_on_ident_patterns() {
        check_assist(
            auto_import,
            r#"
mod foo {
    pub struct Foo {}
}
fn foo() {
    let Foo$0;
}
"#,
            r#"
use foo::Foo;

mod foo {
    pub struct Foo {}
}
fn foo() {
    let Foo;
}
"#,
        );
    }

    #[test]
    fn works_in_derives() {
        check_assist(
            auto_import,
            r#"
//- minicore:derive
mod foo {
    #[rustc_builtin_macro]
    pub macro Copy {}
}
#[derive(Copy$0)]
struct Foo;
"#,
            r#"
use foo::Copy;

mod foo {
    #[rustc_builtin_macro]
    pub macro Copy {}
}
#[derive(Copy)]
struct Foo;
"#,
        );
    }
}
