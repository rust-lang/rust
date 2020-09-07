//! Completes mod declarations.

use base_db::{SourceDatabaseExt, VfsPath};
use hir::{Module, ModuleSource};
use ide_db::RootDatabase;
use rustc_hash::FxHashSet;

use super::{completion_context::CompletionContext, completion_item::Completions};

/// Complete mod declaration, i.e. `mod <|> ;`
pub(super) fn complete_mod(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    let _p = profile::span("completion::complete_mod");

    if !ctx.mod_is_prev {
        return None;
    }

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

    let mod_declaration_candidates = source_root
        .iter()
        .filter(|submodule_candidate_file| submodule_candidate_file != &module_definition_file)
        .filter(|submodule_candidate_file| {
            Some(submodule_candidate_file) != module_declaration_file.as_ref()
        })
        .filter_map(|submodule_file| {
            let submodule_path = source_root.path_for_file(&submodule_file)?;
            if !is_special_rust_file_path(&submodule_path)
                && submodule_path.parent()? == directory_to_look_for_submodules
            {
                submodule_path.file_name_and_extension()
            } else {
                None
            }
        })
        .filter_map(|submodule_file_name_and_extension| match submodule_file_name_and_extension {
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
        })
        .filter(|name| !existing_mod_declarations.contains(name))
        .collect::<Vec<_>>();
    dbg!(mod_declaration_candidates);

    // TODO kb actually add the results

    Some(())
}

fn is_special_rust_file_path(path: &VfsPath) -> bool {
    matches!(
        path.file_name_and_extension(),
        Some(("mod", Some("rs"))) | Some(("lib", Some("rs"))) | Some(("main", Some("rs")))
    )
}

fn directory_to_look_for_submodules(
    module: Module,
    db: &RootDatabase,
    module_file_path: &VfsPath,
) -> Option<VfsPath> {
    let module_directory_path = module_file_path.parent()?;
    let base_directory = if is_special_rust_file_path(module_file_path) {
        Some(module_directory_path)
    } else if let (regular_rust_file_name, Some("rs")) =
        module_file_path.file_name_and_extension()?
    {
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
    } else {
        None
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
