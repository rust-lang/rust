use std::{fmt, iter::FromIterator, sync::Arc};

use hir::MacroFile;
use ra_db::{
    salsa::{
        debug::{DebugQueryTable, TableEntry},
        Database,
    },
    FileTextQuery, SourceRootId,
};
use ra_ide_db::{
    symbol_index::{LibrarySymbolsQuery, SymbolIndex},
    RootDatabase,
};
use ra_prof::{memory_usage, Bytes};
use ra_syntax::{ast, Parse, SyntaxNode};

use crate::FileId;
use rustc_hash::FxHashMap;

fn syntax_tree_stats(db: &RootDatabase) -> SyntaxTreeStats {
    db.query(ra_db::ParseQuery).entries::<SyntaxTreeStats>()
}
fn macro_syntax_tree_stats(db: &RootDatabase) -> SyntaxTreeStats {
    db.query(hir::db::ParseMacroQuery).entries::<SyntaxTreeStats>()
}

// Feature: Status
//
// Shows internal statistic about memory usage of rust-analyzer.
//
// |===
// | Editor  | Action Name
//
// | VS Code | **Rust Analyzer: Status**
// |===
pub(crate) fn status(db: &RootDatabase) -> String {
    let files_stats = db.query(FileTextQuery).entries::<FilesStats>();
    let syntax_tree_stats = syntax_tree_stats(db);
    let macro_syntax_tree_stats = macro_syntax_tree_stats(db);
    let symbols_stats = db.query(LibrarySymbolsQuery).entries::<LibrarySymbolsStats>();
    format!(
        "{}\n{}\n{}\n{} (macros)\n\n\nmemory:\n{}\ngc {:?} seconds ago",
        files_stats,
        symbols_stats,
        syntax_tree_stats,
        macro_syntax_tree_stats,
        memory_usage(),
        db.last_gc.elapsed().as_secs(),
    )
}

#[derive(Default)]
struct FilesStats {
    total: usize,
    size: Bytes,
}

impl fmt::Display for FilesStats {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{} ({}) files", self.total, self.size)
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
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{} trees, {} retained", self.total, self.retained)
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

impl<M> FromIterator<TableEntry<MacroFile, Option<(Parse<SyntaxNode>, M)>>> for SyntaxTreeStats {
    fn from_iter<T>(iter: T) -> SyntaxTreeStats
    where
        T: IntoIterator<Item = TableEntry<MacroFile, Option<(Parse<SyntaxNode>, M)>>>,
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
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{} ({}) symbols", self.total, self.size)
    }
}

impl FromIterator<TableEntry<(), Arc<FxHashMap<SourceRootId, SymbolIndex>>>>
    for LibrarySymbolsStats
{
    fn from_iter<T>(iter: T) -> LibrarySymbolsStats
    where
        T: IntoIterator<Item = TableEntry<(), Arc<FxHashMap<SourceRootId, SymbolIndex>>>>,
    {
        let mut res = LibrarySymbolsStats::default();
        for entry in iter {
            let value = entry.value.unwrap();
            for symbols in value.values() {
                res.total += symbols.len();
                res.size += symbols.memory_size();
            }
        }
        res
    }
}
