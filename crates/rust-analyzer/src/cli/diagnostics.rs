//! Analyze all files in project for diagnostics. Exits with a non-zero status
//! code if any errors are found.

use anyhow::anyhow;
use ra_db::{SourceDatabaseExt, SourceRootId};
use ra_ide::{Analysis, Severity};
use std::{collections::HashSet, path::Path};

use crate::cli::{load_cargo::load_cargo, Result};
use hir::{db::HirDatabase, Crate, Module};

pub fn diagnostics(path: &Path, load_output_dirs: bool) -> Result<()> {
    let (host, roots) = load_cargo(path, load_output_dirs)?;
    let db = host.raw_database();
    let analysis = host.analysis();
    let members = roots
        .into_iter()
        .filter_map(
            |(source_root_id, project_root)| {
                if project_root.is_member() {
                    Some(source_root_id)
                } else {
                    None
                }
            },
        )
        .collect::<HashSet<_>>();

    let mut found_error = false;
    let mut visited_modules = HashSet::new();
    for krate in Crate::all(db) {
        let module = krate.root_module(db).expect("crate without root module");
        check_module(module, db, &mut visited_modules, &members, &analysis, &mut found_error);
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

fn check_module(
    module: Module,
    db: &(impl HirDatabase + SourceDatabaseExt),
    visited_modules: &mut HashSet<Module>,
    members: &HashSet<SourceRootId>,
    analysis: &Analysis,
    found_error: &mut bool,
) {
    let file_id = module.definition_source(db).file_id.original_file(db);
    if !visited_modules.contains(&module) {
        if members.contains(&db.file_source_root(file_id)) {
            println!("processing: {}", db.file_relative_path(file_id));
            for diagnostic in analysis.diagnostics(file_id).unwrap() {
                if matches!(diagnostic.severity, Severity::Error) {
                    *found_error = true;
                }

                println!("{:?}", diagnostic);
            }
        }

        visited_modules.insert(module);

        for child_module in module.children(db) {
            check_module(child_module, db, visited_modules, members, analysis, found_error);
        }
    }
}
