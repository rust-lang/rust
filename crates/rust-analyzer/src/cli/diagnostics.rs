//! Analyze all modules in a project for diagnostics. Exits with a non-zero status
//! code if any errors are found.

use std::path::Path;

use anyhow::anyhow;
use rustc_hash::FxHashSet;

use hir::{db::HirDatabase, Crate, Module};
use ide::{DiagnosticsConfig, Severity};
use ide_db::base_db::SourceDatabaseExt;

use crate::cli::{load_cargo::load_cargo, Result};

fn all_modules(db: &dyn HirDatabase) -> Vec<Module> {
    let mut worklist: Vec<_> =
        Crate::all(db).into_iter().map(|krate| krate.root_module(db)).collect();
    let mut modules = Vec::new();

    while let Some(module) = worklist.pop() {
        modules.push(module);
        worklist.extend(module.children(db));
    }

    modules
}

pub fn diagnostics(path: &Path, load_output_dirs: bool, with_proc_macro: bool) -> Result<()> {
    let (host, _vfs) = load_cargo(path, load_output_dirs, with_proc_macro)?;
    let db = host.raw_database();
    let analysis = host.analysis();

    let mut found_error = false;
    let mut visited_files = FxHashSet::default();

    let work = all_modules(db).into_iter().filter(|module| {
        let file_id = module.definition_source(db).file_id.original_file(db);
        let source_root = db.file_source_root(file_id);
        let source_root = db.source_root(source_root);
        !source_root.is_library
    });

    for module in work {
        let file_id = module.definition_source(db).file_id.original_file(db);
        if !visited_files.contains(&file_id) {
            let crate_name =
                module.krate().display_name(db).as_deref().unwrap_or("unknown").to_string();
            println!("processing crate: {}, module: {}", crate_name, _vfs.file_path(file_id));
            for diagnostic in analysis.diagnostics(&DiagnosticsConfig::default(), file_id).unwrap()
            {
                if matches!(diagnostic.severity, Severity::Error) {
                    found_error = true;
                }

                println!("{:?}", diagnostic);
            }

            visited_files.insert(file_id);
        }
    }

    println!();
    println!("diagnostic scan complete");

    if found_error {
        println!();
        Err(anyhow!("diagnostic error detected"))
    } else {
        Ok(())
    }
}
