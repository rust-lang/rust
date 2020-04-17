//! Analyze all modules in a project for diagnostics. Exits with a non-zero status
//! code if any errors are found.

use anyhow::anyhow;
use ra_db::SourceDatabaseExt;
use ra_ide::Severity;
use std::{collections::HashSet, path::Path};

use crate::cli::{load_cargo::load_cargo, Result};
use hir::Semantics;

pub fn diagnostics(
    path: &Path,
    load_output_dirs: bool,
    with_proc_macro: bool,
    all: bool,
) -> Result<()> {
    let (host, roots) = load_cargo(path, load_output_dirs, with_proc_macro)?;
    let db = host.raw_database();
    let analysis = host.analysis();
    let semantics = Semantics::new(db);
    let members = roots
        .into_iter()
        .filter_map(|(source_root_id, project_root)| {
            // filter out dependencies
            if project_root.is_member() {
                Some(source_root_id)
            } else {
                None
            }
        })
        .collect::<HashSet<_>>();

    let mut found_error = false;
    let mut visited_files = HashSet::new();
    for source_root_id in members {
        for file_id in db.source_root(source_root_id).walk() {
            // Filter out files which are not actually modules (unless `--all` flag is
            // passed). In the rust-analyzer repository this filters out the parser test files.
            if semantics.to_module_def(file_id).is_some() || all {
                if !visited_files.contains(&file_id) {
                    let crate_name = if let Some(module) = semantics.to_module_def(file_id) {
                        if let Some(name) = module.krate().display_name(db) {
                            format!("{}", name)
                        } else {
                            String::from("unknown")
                        }
                    } else {
                        String::from("unknown")
                    };
                    println!(
                        "processing crate: {}, module: {}",
                        crate_name,
                        db.file_relative_path(file_id)
                    );
                    for diagnostic in analysis.diagnostics(file_id).unwrap() {
                        if matches!(diagnostic.severity, Severity::Error) {
                            found_error = true;
                        }

                        println!("{:?}", diagnostic);
                    }

                    visited_files.insert(file_id);
                }
            }
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
