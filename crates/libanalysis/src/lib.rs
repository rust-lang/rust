extern crate parking_lot;
#[macro_use]
extern crate log;
extern crate once_cell;
extern crate libsyntax2;
extern crate libeditor;
extern crate fst;
extern crate rayon;
extern crate relative_path;

mod symbol_index;
mod module_map;
mod imp;

use std::sync::Arc;

use relative_path::{RelativePath, RelativePathBuf};
use libsyntax2::{File, TextRange, TextUnit, AtomEdit};
use imp::{AnalysisImpl, AnalysisHostImpl};

pub use libeditor::{
    StructureNode, LineIndex, FileSymbol,
    Runnable, RunnableKind, HighlightedRange, CompletionItem,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId(pub u32);

pub trait FileResolver: Send + Sync + 'static {
    fn file_stem(&self, id: FileId) -> String;
    fn resolve(&self, id: FileId, path: &RelativePath) -> Option<FileId>;
}

#[derive(Debug)]
pub struct AnalysisHost {
    imp: AnalysisHostImpl
}

impl AnalysisHost {
    pub fn new() -> AnalysisHost {
        AnalysisHost { imp: AnalysisHostImpl::new() }
    }
    pub fn analysis(&self, file_resolver: impl FileResolver) -> Analysis {
        Analysis { imp: self.imp.analysis(Arc::new(file_resolver)) }
    }
    pub fn change_file(&mut self, file_id: FileId, text: Option<String>) {
        self.change_files(::std::iter::once((file_id, text)));
    }
    pub fn change_files(&mut self, mut changes: impl Iterator<Item=(FileId, Option<String>)>) {
        self.imp.change_files(&mut changes)
    }
}

#[derive(Debug)]
pub struct SourceChange {
    pub label: String,
    pub source_file_edits: Vec<SourceFileEdit>,
    pub file_system_edits: Vec<FileSystemEdit>,
    pub cursor_position: Option<Position>,
}

#[derive(Debug)]
pub struct Position {
    pub file_id: FileId,
    pub offset: TextUnit,
}

#[derive(Debug)]
pub struct SourceFileEdit {
    pub file_id: FileId,
    pub edits: Vec<AtomEdit>,
}

#[derive(Debug)]
pub enum FileSystemEdit {
    CreateFile {
        anchor: FileId,
        path: RelativePathBuf,
    },
    MoveFile {
        file: FileId,
        path: RelativePathBuf,
    }
}

#[derive(Debug)]
pub struct Diagnostic {
    pub message: String,
    pub range: TextRange,
    pub fix: Option<SourceChange>,
}

#[derive(Debug)]
pub struct Query {
    query: String,
    lowercased: String,
    only_types: bool,
    exact: bool,
    limit: usize,
}

impl Query {
    pub fn new(query: String) -> Query {
        let lowercased = query.to_lowercase();
        Query {
            query,
            lowercased,
            only_types: false,
            exact: false,
            limit: usize::max_value()
        }
    }
    pub fn only_types(&mut self) {
        self.only_types = true;
    }
    pub fn exact(&mut self) {
        self.exact = true;
    }
    pub fn limit(&mut self, limit: usize) {
        self.limit = limit
    }
}

#[derive(Clone, Debug)]
pub struct Analysis {
    imp: AnalysisImpl
}

impl Analysis {
    pub fn file_syntax(&self, file_id: FileId) -> File {
        self.imp.file_syntax(file_id)
    }
    pub fn file_line_index(&self, file_id: FileId) -> LineIndex {
        self.imp.file_line_index(file_id)
    }
    pub fn extend_selection(&self, file: &File, range: TextRange) -> TextRange {
        libeditor::extend_selection(file, range).unwrap_or(range)
    }
    pub fn matching_brace(&self, file: &File, offset: TextUnit) -> Option<TextUnit> {
        libeditor::matching_brace(file, offset)
    }
    pub fn syntax_tree(&self, file_id: FileId) -> String {
        let file = self.file_syntax(file_id);
        libeditor::syntax_tree(&file)
    }
    pub fn join_lines(&self, file_id: FileId, range: TextRange) -> SourceChange {
        let file = self.file_syntax(file_id);
        SourceChange::from_local_edit(file_id, "join lines", libeditor::join_lines(&file, range))
    }
    pub fn on_eq_typed(&self, file_id: FileId, offset: TextUnit) -> Option<SourceChange> {
        let file = self.file_syntax(file_id);
        Some(SourceChange::from_local_edit(file_id, "add semicolon", libeditor::on_eq_typed(&file, offset)?))
    }
    pub fn file_structure(&self, file_id: FileId) -> Vec<StructureNode> {
        let file = self.file_syntax(file_id);
        libeditor::file_structure(&file)
    }
    pub fn symbol_search(&self, query: Query) -> Vec<(FileId, FileSymbol)> {
        self.imp.world_symbols(query)
    }
    pub fn approximately_resolve_symbol(&self, file_id: FileId, offset: TextUnit) -> Vec<(FileId, FileSymbol)> {
        self.imp.approximately_resolve_symbol(file_id, offset)
    }
    pub fn parent_module(&self, file_id: FileId) -> Vec<(FileId, FileSymbol)> {
        self.imp.parent_module(file_id)
    }
    pub fn runnables(&self, file_id: FileId) -> Vec<Runnable> {
        let file = self.file_syntax(file_id);
        libeditor::runnables(&file)
    }
    pub fn highlight(&self, file_id: FileId) -> Vec<HighlightedRange> {
        let file = self.file_syntax(file_id);
        libeditor::highlight(&file)
    }
    pub fn completions(&self, file_id: FileId, offset: TextUnit) -> Option<Vec<CompletionItem>> {
        let file = self.file_syntax(file_id);
        libeditor::scope_completion(&file, offset)
    }
    pub fn assists(&self, file_id: FileId, offset: TextUnit) -> Vec<SourceChange> {
        self.imp.assists(file_id, offset)
    }
    pub fn diagnostics(&self, file_id: FileId) -> Vec<Diagnostic> {
        self.imp.diagnostics(file_id)
    }
}
