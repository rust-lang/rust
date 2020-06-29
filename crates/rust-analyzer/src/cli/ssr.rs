//! Applies structured search replace rules from the command line.

use crate::cli::{load_cargo::load_cargo, Result};
use ra_ide::SourceFileEdit;
use ra_ssr::{MatchFinder, SsrRule};

pub fn apply_ssr_rules(rules: Vec<SsrRule>) -> Result<()> {
    use ra_db::SourceDatabaseExt;
    use ra_ide_db::symbol_index::SymbolsDatabase;
    let (host, vfs) = load_cargo(&std::env::current_dir()?, true, true)?;
    let db = host.raw_database();
    let mut match_finder = MatchFinder::new(db);
    for rule in rules {
        match_finder.add_rule(rule);
    }
    let mut edits = Vec::new();
    for &root in db.local_roots().iter() {
        let sr = db.source_root(root);
        for file_id in sr.iter() {
            if let Some(edit) = match_finder.edits_for_file(file_id) {
                edits.push(SourceFileEdit { file_id, edit });
            }
        }
    }
    for edit in edits {
        if let Some(path) = vfs.file_path(edit.file_id).as_path() {
            let mut contents = db.file_text(edit.file_id).to_string();
            edit.edit.apply(&mut contents);
            std::fs::write(path, contents)?;
        }
    }
    Ok(())
}
