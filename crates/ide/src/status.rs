use std::{fmt, iter::FromIterator, sync::Arc};

use hir::{ExpandResult, MacroFile};
use ide_db::base_db::{
    salsa::debug::{DebugQueryTable, TableEntry},
    CrateId, FileId, FileTextQuery, SourceDatabase, SourceRootId,
};
use ide_db::{
    symbol_index::{LibrarySymbolsQuery, SymbolIndex},
    RootDatabase,
};
use itertools::Itertools;
use profile::{memory_usage, Bytes};
use std::env;
use stdx::format_to;
use syntax::{ast, Parse, SyntaxNode};

fn syntax_tree_stats(db: &RootDatabase) -> SyntaxTreeStats {
    ide_db::base_db::ParseQuery.in_db(db).entries::<SyntaxTreeStats>()
}
fn macro_syntax_tree_stats(db: &RootDatabase) -> SyntaxTreeStats {
    hir::db::ParseMacroExpansionQuery.in_db(db).entries::<SyntaxTreeStats>()
}

// Feature: Status
//
// Shows internal statistic about memory usage of rust-analyzer.
//
// |===
// | Editor  | Action Name
//
// | VS Code | **rust-analyzer: Status**
// |===
// image::https://user-images.githubusercontent.com/48062697/113065584-05f34500-91b1-11eb-98cc-5c196f76be7f.gif[]
pub(crate) fn status(db: &RootDatabase, file_id: Option<FileId>) -> String {
    let mut buf = String::new();
    format_to!(buf, "{}\n", FileTextQuery.in_db(db).entries::<FilesStats>());
    format_to!(buf, "{}\n", LibrarySymbolsQuery.in_db(db).entries::<LibrarySymbolsStats>());
    format_to!(buf, "{}\n", syntax_tree_stats(db));
    format_to!(buf, "{} (Macros)\n", macro_syntax_tree_stats(db));
    format_to!(buf, "{} in total\n", memory_usage());
    if env::var("RA_COUNT").is_ok() {
        format_to!(buf, "\nCounts:\n{}", profile::countme::get_all());
    }

    if let Some(file_id) = file_id {
        format_to!(buf, "\nFile info:\n");
        let crates = crate::parent_module::crate_for(db, file_id);
        if crates.is_empty() {
            format_to!(buf, "Does not belong to any crate");
        }
        let crate_graph = db.crate_graph();
        for krate in crates {
            let display_crate = |krate: CrateId| match &crate_graph[krate].display_name {
                Some(it) => format!("{}({:?})", it, krate),
                None => format!("{:?}", krate),
            };
            format_to!(buf, "Crate: {}\n", display_crate(krate));
            let deps = crate_graph[krate]
                .dependencies
                .iter()
                .map(|dep| format!("{}={:?}", dep.name, dep.crate_id))
                .format(", ");
            format_to!(buf, "Dependencies: {}\n", deps);
        }
    }

    buf.trim().to_string()
}

#[derive(Default)]
struct FilesStats {
    total: usize,
    size: Bytes,
}

impl fmt::Display for FilesStats {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{} of files", self.size)
    }
}

impl FromIterator<TableEntry<FileId, Arc<String>>> for FilesStats {
    fn from_iter<T>(iter: T) -> FilesStats
    where
        T: IntoIterator<Item = TableEntry<FileId, Arc<String>>>,
    {
        let mut res = FilesStats::default();
        for entry in iter {
            res.total += 1;
            res.size += entry.value.unwrap().len();
        }
        res
    }
}

#[derive(Default)]
pub(crate) struct SyntaxTreeStats {
    total: usize,
    pub(crate) retained: usize,
}

impl fmt::Display for SyntaxTreeStats {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{} trees, {} preserved", self.total, self.retained)
    }
}

impl FromIterator<TableEntry<FileId, Parse<ast::SourceFile>>> for SyntaxTreeStats {
    fn from_iter<T>(iter: T) -> SyntaxTreeStats
    where
        T: IntoIterator<Item = TableEntry<FileId, Parse<ast::SourceFile>>>,
    {
        let mut res = SyntaxTreeStats::default();
        for entry in iter {
            res.total += 1;
            res.retained += entry.value.is_some() as usize;
        }
        res
    }
}

impl<M> FromIterator<TableEntry<MacroFile, ExpandResult<Option<(Parse<SyntaxNode>, M)>>>>
    for SyntaxTreeStats
{
    fn from_iter<T>(iter: T) -> SyntaxTreeStats
    where
        T: IntoIterator<Item = TableEntry<MacroFile, ExpandResult<Option<(Parse<SyntaxNode>, M)>>>>,
    {
        let mut res = SyntaxTreeStats::default();
        for entry in iter {
            res.total += 1;
            res.retained += entry.value.is_some() as usize;
        }
        res
    }
}

#[derive(Default)]
struct LibrarySymbolsStats {
    total: usize,
    size: Bytes,
}

impl fmt::Display for LibrarySymbolsStats {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{} of index symbols ({})", self.size, self.total)
    }
}

impl FromIterator<TableEntry<SourceRootId, Arc<SymbolIndex>>> for LibrarySymbolsStats {
    fn from_iter<T>(iter: T) -> LibrarySymbolsStats
    where
        T: IntoIterator<Item = TableEntry<SourceRootId, Arc<SymbolIndex>>>,
    {
        let mut res = LibrarySymbolsStats::default();
        for entry in iter {
            let symbols = entry.value.unwrap();
            res.total += symbols.len();
            res.size += symbols.memory_size();
        }
        res
    }
}
