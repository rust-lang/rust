//! Applies structured search replace rules from the command line.

use ide_ssr::MatchFinder;
use project_model::CargoConfig;

use crate::cli::{
    flags,
    load_cargo::{load_workspace_at, LoadCargoConfig, ProcMacroServerChoice},
    Result,
};

impl flags::Ssr {
    pub fn run(self) -> Result<()> {
        use ide_db::base_db::SourceDatabaseExt;
        let cargo_config = CargoConfig::default();
        let load_cargo_config = LoadCargoConfig {
            load_out_dirs_from_check: true,
            with_proc_macro_server: ProcMacroServerChoice::Sysroot,
            prefill_caches: false,
        };
        let (host, vfs, _proc_macro) = load_workspace_at(
            &std::env::current_dir()?,
            &cargo_config,
            &load_cargo_config,
            &|_| {},
        )?;
        let db = host.raw_database();
        let mut match_finder = MatchFinder::at_first_file(db)?;
        for rule in self.rule {
            match_finder.add_rule(rule)?;
        }
        let edits = match_finder.edits();
        for (file_id, edit) in edits {
            if let Some(path) = vfs.file_path(file_id).as_path() {
                let mut contents = db.file_text(file_id).to_string();
                edit.apply(&mut contents);
                std::fs::write(path, contents)?;
            }
        }
        Ok(())
    }
}

impl flags::Search {
    /// Searches for `patterns`, printing debug information for any nodes whose text exactly matches
    /// `debug_snippet`. This is intended for debugging and probably isn't in it's current form useful
    /// for much else.
    pub fn run(self) -> Result<()> {
        use ide_db::base_db::SourceDatabaseExt;
        use ide_db::symbol_index::SymbolsDatabase;
        let cargo_config = CargoConfig::default();
        let load_cargo_config = LoadCargoConfig {
            load_out_dirs_from_check: true,
            with_proc_macro_server: ProcMacroServerChoice::Sysroot,
            prefill_caches: false,
        };
        let (host, _vfs, _proc_macro) = load_workspace_at(
            &std::env::current_dir()?,
            &cargo_config,
            &load_cargo_config,
            &|_| {},
        )?;
        let db = host.raw_database();
        let mut match_finder = MatchFinder::at_first_file(db)?;
        for pattern in self.pattern {
            match_finder.add_search_pattern(pattern)?;
        }
        if let Some(debug_snippet) = &self.debug {
            for &root in db.local_roots().iter() {
                let sr = db.source_root(root);
                for file_id in sr.iter() {
                    for debug_info in match_finder.debug_where_text_equal(file_id, debug_snippet) {
                        println!("{debug_info:#?}");
                    }
                }
            }
        } else {
            for m in match_finder.matches().flattened().matches {
                // We could possibly at some point do something more useful than just printing
                // the matched text. For now though, that's the easiest thing to do.
                println!("{}", m.matched_text());
            }
        }
        Ok(())
    }
}
