//! Applies structured search replace rules from the command line.

use crate::cli::{load_cargo::load_cargo, Result};
use ra_ide::SourceFileEdit;
use ra_ssr::{MatchFinder, SsrPattern, SsrRule};

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

/// Searches for `patterns`, printing debug information for any nodes whose text exactly matches
/// `debug_snippet`. This is intended for debugging and probably isn't in it's current form useful
/// for much else.
pub fn search_for_patterns(patterns: Vec<SsrPattern>, debug_snippet: Option<String>) -> Result<()> {
    use ra_db::SourceDatabaseExt;
    use ra_ide_db::symbol_index::SymbolsDatabase;
    let (host, vfs) = load_cargo(&std::env::current_dir()?, true, true)?;
    let db = host.raw_database();
    let mut match_finder = MatchFinder::new(db);
    for pattern in patterns {
        match_finder.add_search_pattern(pattern);
    }
    for &root in db.local_roots().iter() {
        let sr = db.source_root(root);
        for file_id in sr.iter() {
            if let Some(debug_snippet) = &debug_snippet {
                for debug_info in match_finder.debug_where_text_equal(file_id, debug_snippet) {
                    println!("{:#?}", debug_info);
                }
            } else {
                let matches = match_finder.find_matches_in_file(file_id);
                if !matches.matches.is_empty() {
                    let matches = matches.flattened().matches;
                    if let Some(path) = vfs.file_path(file_id).as_path() {
                        println!("{} matches in '{}'", matches.len(), path.to_string_lossy());
                    }
                    // We could possibly at some point do something more useful than just printing
                    // the matched text. For now though, that's the easiest thing to do.
                    for m in matches {
                        println!("{}", m.matched_text());
                    }
                }
            }
        }
    }
    Ok(())
}
