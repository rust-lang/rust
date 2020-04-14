//! Analyze all modules in a project for diagnostics. Exits with a non-zero status
//! code if any errors are found.

use anyhow::anyhow;
use ra_db::{SourceDatabase, SourceDatabaseExt};
use ra_ide::Severity;
use std::{collections::HashSet, path::Path};

use crate::cli::{load_cargo::load_cargo, Result};
use hir::Semantics;

pub fn diagnostics(path: &Path, load_output_dirs: bool, all: bool) -> Result<()> {
    let (host, roots) = load_cargo(path, load_output_dirs)?;
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
    let crate_graph = db.crate_graph();
    for crate_id in crate_graph.iter() {
        let krate = &crate_graph[crate_id];
        if let Some(crate_name) = &krate.display_name {
            println!("processing crate: {}", crate_name);
        } else {
            println!("processing crate: unknown");
        }
        for file_id in db.source_root(db.file_source_root(krate.root_file_id)).walk() {
            // Filter out files which are not actually modules (unless `--all` flag is
            // passed). In the rust-analyzer repository this filters out the parser test files.
            if semantics.to_module_def(file_id).is_some() || all {
                if !visited_files.contains(&file_id) {
                    if members.contains(&db.file_source_root(file_id)) {
                        println!("processing module: {}", db.file_relative_path(file_id));
                        for diagnostic in analysis.diagnostics(file_id).unwrap() {
                            if matches!(diagnostic.severity, Severity::Error) {
                                found_error = true;
                            }

                            println!("{:?}", diagnostic);
                        }
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
