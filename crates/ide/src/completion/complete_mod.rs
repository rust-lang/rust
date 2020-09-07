//! Completes mod declarations.

use base_db::{SourceDatabaseExt, VfsPath};
use hir::{Module, ModuleSource};
use ide_db::RootDatabase;

use super::{completion_context::CompletionContext, completion_item::Completions};

/// Complete mod declaration, i.e. `mod <|> ;`
pub(super) fn complete_mod(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    let current_module = ctx.scope.module()?;

    // TODO kb filter out declarations in possible_sudmobule_names
    // let declaration_source = current_module.declaration_source(ctx.db);
    let module_definition_source_file =
        current_module.definition_source(ctx.db).file_id.original_file(ctx.db);
    let source_root = ctx.db.source_root(ctx.db.file_source_root(module_definition_source_file));
    let directory_to_look_for_submodules = directory_to_look_for_submodules(
        current_module,
        ctx.db,
        source_root.path_for_file(&module_definition_source_file)?,
    )?;

    let mod_declaration_candidates = source_root
        .iter()
        .filter(|submodule_file| submodule_file != &module_definition_source_file)
        .filter_map(|submodule_file| {
            let submodule_path = source_root.path_for_file(&submodule_file)?;
            if submodule_path.parent()? == directory_to_look_for_submodules {
                submodule_path.file_name_and_extension()
            } else {
                None
            }
        })
        .filter_map(|file_name_and_extension| {
            match file_name_and_extension {
                // TODO kb in src/bin when a module is included into another,
                // the included file gets "moved" into a directory below and now cannot add any other modules
                ("mod", Some("rs")) | ("lib", Some("rs")) | ("main", Some("rs")) => None,
                (file_name, Some("rs")) => Some(file_name.to_owned()),
                (subdirectory_name, None) => {
                    let mod_rs_path =
                        directory_to_look_for_submodules.join(subdirectory_name)?.join("mod.rs")?;
                    if source_root.file_for_path(&mod_rs_path).is_some() {
                        Some(subdirectory_name.to_owned())
                    } else {
                        None
                    }
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    dbg!(mod_declaration_candidates);

    // TODO kb exlude existing children from the candidates
    let existing_children = current_module.children(ctx.db).collect::<Vec<_>>();

    Some(())
}

fn directory_to_look_for_submodules(
    module: Module,
    db: &RootDatabase,
    module_file_path: &VfsPath,
) -> Option<VfsPath> {
    let module_directory_path = module_file_path.parent()?;

    let base_directory = match module_file_path.file_name_and_extension()? {
        ("mod", Some("rs")) | ("lib", Some("rs")) | ("main", Some("rs")) => {
            Some(module_directory_path)
        }
        (regular_rust_file_name, Some("rs")) => {
            if matches!(
                (
                    module_directory_path
                        .parent()
                        .as_ref()
                        .and_then(|path| path.file_name_and_extension()),
                    module_directory_path.file_name_and_extension(),
                ),
                (Some(("src", None)), Some(("bin", None)))
            ) {
                // files in /src/bin/ can import each other directly
                Some(module_directory_path)
            } else {
                module_directory_path.join(regular_rust_file_name)
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
