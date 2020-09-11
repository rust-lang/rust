//! Completes mod declarations.

use base_db::{SourceDatabaseExt, VfsPath};
use hir::{Module, ModuleSource};
use ide_db::RootDatabase;
use rustc_hash::FxHashSet;

use crate::{CompletionItem, CompletionItemKind};

use super::{
    completion_context::CompletionContext, completion_item::CompletionKind,
    completion_item::Completions,
};

/// Complete mod declaration, i.e. `mod <|> ;`
pub(super) fn complete_mod(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    let mod_under_caret = match &ctx.mod_declaration_under_caret {
        Some(mod_under_caret) if mod_under_caret.item_list().is_some() => return None,
        Some(mod_under_caret) => mod_under_caret,
        None => return None,
    };

    let _p = profile::span("completion::complete_mod");

    let current_module = ctx.scope.module()?;

    let module_definition_file =
        current_module.definition_source(ctx.db).file_id.original_file(ctx.db);
    let source_root = ctx.db.source_root(ctx.db.file_source_root(module_definition_file));
    let directory_to_look_for_submodules = directory_to_look_for_submodules(
        current_module,
        ctx.db,
        source_root.path_for_file(&module_definition_file)?,
    )?;

    let existing_mod_declarations = current_module
        .children(ctx.db)
        .filter_map(|module| Some(module.name(ctx.db)?.to_string()))
        .collect::<FxHashSet<_>>();

    let module_declaration_file =
        current_module.declaration_source(ctx.db).map(|module_declaration_source_file| {
            module_declaration_source_file.file_id.original_file(ctx.db)
        });

    source_root
        .iter()
        .filter(|submodule_candidate_file| submodule_candidate_file != &module_definition_file)
        .filter(|submodule_candidate_file| {
            Some(submodule_candidate_file) != module_declaration_file.as_ref()
        })
        .filter_map(|submodule_file| {
            let submodule_path = source_root.path_for_file(&submodule_file)?;
            let directory_with_submodule = submodule_path.parent()?;
            match submodule_path.name_and_extension()? {
                ("lib", Some("rs")) | ("main", Some("rs")) => None,
                ("mod", Some("rs")) => {
                    if directory_with_submodule.parent()? == directory_to_look_for_submodules {
                        match directory_with_submodule.name_and_extension()? {
                            (directory_name, None) => Some(directory_name.to_owned()),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                (file_name, Some("rs"))
                    if directory_with_submodule == directory_to_look_for_submodules =>
                {
                    Some(file_name.to_owned())
                }
                _ => None,
            }
        })
        .filter(|name| !existing_mod_declarations.contains(name))
        .for_each(|submodule_name| {
            let mut label = submodule_name;
            if mod_under_caret.semicolon_token().is_none() {
                label.push(';')
            }
            acc.add(
                CompletionItem::new(CompletionKind::Magic, ctx.source_range(), &label)
                    .kind(CompletionItemKind::Module),
            )
        });

    Some(())
}

fn directory_to_look_for_submodules(
    module: Module,
    db: &RootDatabase,
    module_file_path: &VfsPath,
) -> Option<VfsPath> {
    let directory_with_module_path = module_file_path.parent()?;
    let base_directory = match module_file_path.name_and_extension()? {
        ("mod", Some("rs")) | ("lib", Some("rs")) | ("main", Some("rs")) => {
            Some(directory_with_module_path)
        }
        (regular_rust_file_name, Some("rs")) => {
            if matches!(
                (
                    directory_with_module_path
                        .parent()
                        .as_ref()
                        .and_then(|path| path.name_and_extension()),
                    directory_with_module_path.name_and_extension(),
                ),
                (Some(("src", None)), Some(("bin", None)))
            ) {
                // files in /src/bin/ can import each other directly
                Some(directory_with_module_path)
            } else {
                directory_with_module_path.join(regular_rust_file_name)
            }
        }
        _ => None,
    }?;

    let mut resulting_path = base_directory;
    for module in module_chain_to_containing_module_file(module, db) {
        if let Some(name) = module.name(db) {
            resulting_path = resulting_path.join(&name.to_string())?;
        }
    }

    Some(resulting_path)
}

fn module_chain_to_containing_module_file(
    current_module: Module,
    db: &RootDatabase,
) -> Vec<Module> {
    let mut path = Vec::new();

    let mut current_module = Some(current_module);
    while let Some(ModuleSource::Module(_)) =
        current_module.map(|module| module.definition_source(db).value)
    {
        if let Some(module) = current_module {
            path.insert(0, module);
            current_module = module.parent(db);
        } else {
            current_module = None;
        }
    }

    path
}

#[cfg(test)]
mod tests {
    use crate::completion::{test_utils::completion_list, CompletionKind};
    use expect_test::{expect, Expect};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Magic);
        expect.assert_eq(&actual);
    }

    #[test]
    fn lib_module_completion() {
        check(
            r#"
            //- /lib.rs
            mod <|>
            //- /foo.rs
            fn foo() {}
            //- /foo/ignored_foo.rs
            fn ignored_foo() {}
            //- /bar/mod.rs
            fn bar() {}
            //- /bar/ignored_bar.rs
            fn ignored_bar() {}
        "#,
            expect![[r#"
                md bar;
                md foo;
            "#]],
        );
    }

    #[test]
    fn no_module_completion_with_module_body() {
        check(
            r#"
            //- /lib.rs
            mod <|> {

            }
            //- /foo.rs
            fn foo() {}
        "#,
            expect![[r#""#]],
        );
    }

    #[test]
    fn main_module_completion() {
        check(
            r#"
            //- /main.rs
            mod <|>
            //- /foo.rs
            fn foo() {}
            //- /foo/ignored_foo.rs
            fn ignored_foo() {}
            //- /bar/mod.rs
            fn bar() {}
            //- /bar/ignored_bar.rs
            fn ignored_bar() {}
        "#,
            expect![[r#"
                md bar;
                md foo;
            "#]],
        );
    }

    #[test]
    fn main_test_module_completion() {
        check(
            r#"
            //- /main.rs
            mod tests {
                mod <|>;
            }
            //- /tests/foo.rs
            fn foo() {}
        "#,
            expect![[r#"
                md foo
            "#]],
        );
    }

    #[test]
    fn directly_nested_module_completion() {
        check(
            r#"
            //- /lib.rs
            mod foo;
            //- /foo.rs
            mod <|>;
            //- /foo/bar.rs
            fn bar() {}
            //- /foo/bar/ignored_bar.rs
            fn ignored_bar() {}
            //- /foo/baz/mod.rs
            fn baz() {}
            //- /foo/moar/ignored_moar.rs
            fn ignored_moar() {}
        "#,
            expect![[r#"
                md bar
                md baz
            "#]],
        );
    }

    #[test]
    fn nested_in_source_module_completion() {
        check(
            r#"
            //- /lib.rs
            mod foo;
            //- /foo.rs
            mod bar {
                mod <|>
            }
            //- /foo/bar/baz.rs
            fn baz() {}
        "#,
            expect![[r#"
                md baz;
            "#]],
        );
    }

    // FIXME binary modules are not supported in tests properly
    // Binary modules are a bit special, they allow importing the modules from `/src/bin`
    // and that's why are good to test two things:
    // * no cycles are allowed in mod declarations
    // * no modules from the parent directory are proposed
    // Unfortunately, binary modules support is in cargo not rustc,
    // hence the test does not work now
    //
    // #[test]
    // fn regular_bin_module_completion() {
    //     check(
    //         r#"
    //         //- /src/bin.rs
    //         fn main() {}
    //         //- /src/bin/foo.rs
    //         mod <|>
    //         //- /src/bin/bar.rs
    //         fn bar() {}
    //         //- /src/bin/bar/bar_ignored.rs
    //         fn bar_ignored() {}
    //     "#,
    //         expect![[r#"
    //             md bar;
    //         "#]],
    //     );
    // }

    #[test]
    fn already_declared_bin_module_completion_omitted() {
        check(
            r#"
            //- /src/bin.rs
            fn main() {}
            //- /src/bin/foo.rs
            mod <|>
            //- /src/bin/bar.rs
            mod foo;
            fn bar() {}
            //- /src/bin/bar/bar_ignored.rs
            fn bar_ignored() {}
        "#,
            expect![[r#""#]],
        );
    }
}
