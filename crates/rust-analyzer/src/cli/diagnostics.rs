//! Analyze all modules in a project for diagnostics. Exits with a non-zero status
//! code if any errors are found.

use std::path::Path;

use anyhow::anyhow;
use rustc_hash::FxHashSet;

use hir::Crate;
use ra_db::SourceDatabaseExt;
use ra_ide::Severity;

use crate::cli::{load_cargo::load_cargo, Result};

pub fn diagnostics(
    path: &Path,
    load_output_dirs: bool,
    with_proc_macro: bool,
    _all: bool,
) -> Result<()> {
    let (host, _vfs) = load_cargo(path, load_output_dirs, with_proc_macro)?;
    let db = host.raw_database();
    let analysis = host.analysis();

    let mut found_error = false;
    let mut visited_files = FxHashSet::default();

    let mut work = Vec::new();
    let krates = Crate::all(db);
    for krate in krates {
        let module = krate.root_module(db).expect("crate without root module");
        let file_id = module.definition_source(db).file_id;
        let file_id = file_id.original_file(db);
        let source_root = db.file_source_root(file_id);
        let source_root = db.source_root(source_root);
        if !source_root.is_library {
            work.push(module);
        }
    }

    for module in work {
        let file_id = module.definition_source(db).file_id.original_file(db);
        if !visited_files.contains(&file_id) {
            let crate_name = if let Some(name) = module.krate().display_name(db) {
                format!("{}", name)
            } else {
                String::from("unknown")
            };
            println!("processing crate: {}, module: {}", crate_name, _vfs.file_path(file_id));
            for diagnostic in analysis.diagnostics(file_id).unwrap() {
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
