extern crate parking_lot;
#[macro_use]
extern crate log;
extern crate once_cell;
extern crate libsyntax2;
extern crate libeditor;
extern crate fst;
extern crate rayon;
extern crate relative_path;
#[macro_use]
extern crate crossbeam_channel;

mod symbol_index;
mod module_map;
mod imp;
mod job;
mod roots;

use std::{
    sync::Arc,
    collections::HashMap,
    fmt::Debug,
};

use relative_path::{RelativePath, RelativePathBuf};
use libsyntax2::{File, TextRange, TextUnit, AtomEdit};
use imp::{AnalysisImpl, AnalysisHostImpl, FileResolverImp};

pub use libeditor::{
    StructureNode, LineIndex, FileSymbol,
    Runnable, RunnableKind, HighlightedRange, CompletionItem,
};
pub use job::{JobToken, JobHandle};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CrateId(pub u32);

#[derive(Debug, Clone, Default)]
pub struct CrateGraph {
    pub crate_roots: HashMap<CrateId, FileId>,
}

pub trait FileResolver: Debug + Send + Sync + 'static {
    fn file_stem(&self, file_id: FileId) -> String;
    fn resolve(&self, file_id: FileId, path: &RelativePath) -> Option<FileId>;
}

#[derive(Debug)]
pub struct AnalysisHost {
    imp: AnalysisHostImpl
}

impl AnalysisHost {
    pub fn new() -> AnalysisHost {
        AnalysisHost { imp: AnalysisHostImpl::new() }
    }
    pub fn analysis(&self) -> Analysis {
        Analysis { imp: self.imp.analysis() }
    }
    pub fn change_file(&mut self, file_id: FileId, text: Option<String>) {
        self.change_files(::std::iter::once((file_id, text)));
    }
    pub fn change_files(&mut self, mut changes: impl Iterator<Item=(FileId, Option<String>)>) {
        self.imp.change_files(&mut changes)
    }
    pub fn set_file_resolver(&mut self, resolver: Arc<FileResolver>) {
        self.imp.set_file_resolver(FileResolverImp::new(resolver));
    }
    pub fn set_crate_graph(&mut self, graph: CrateGraph) {
        self.imp.set_crate_graph(graph)
    }
    pub fn add_library(&mut self, data: LibraryData) {
        self.imp.add_library(data.root)
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
    libs: bool,
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
            libs: false,
            exact: false,
            limit: usize::max_value()
        }
    }
    pub fn only_types(&mut self) {
        self.only_types = true;
    }
    pub fn libs(&mut self) {
        self.libs = true;
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
        self.imp.file_syntax(file_id).clone()
    }
    pub fn file_line_index(&self, file_id: FileId) -> LineIndex {
        self.imp.file_line_index(file_id).clone()
    }
    pub fn extend_selection(&self, file: &File, range: TextRange) -> TextRange {
        libeditor::extend_selection(file, range).unwrap_or(range)
    }
    pub fn matching_brace(&self, file: &File, offset: TextUnit) -> Option<TextUnit> {
        libeditor::matching_brace(file, offset)
    }
    pub fn syntax_tree(&self, file_id: FileId) -> String {
        let file = self.imp.file_syntax(file_id);
        libeditor::syntax_tree(file)
    }
    pub fn join_lines(&self, file_id: FileId, range: TextRange) -> SourceChange {
        let file = self.imp.file_syntax(file_id);
        SourceChange::from_local_edit(file_id, "join lines", libeditor::join_lines(file, range))
    }
    pub fn on_eq_typed(&self, file_id: FileId, offset: TextUnit) -> Option<SourceChange> {
        let file = self.imp.file_syntax(file_id);
        Some(SourceChange::from_local_edit(file_id, "add semicolon", libeditor::on_eq_typed(file, offset)?))
    }
    pub fn file_structure(&self, file_id: FileId) -> Vec<StructureNode> {
        let file = self.imp.file_syntax(file_id);
        libeditor::file_structure(file)
    }
    pub fn symbol_search(&self, query: Query, token: &JobToken) -> Vec<(FileId, FileSymbol)> {
        self.imp.world_symbols(query, token)
    }
    pub fn approximately_resolve_symbol(&self, file_id: FileId, offset: TextUnit, token: &JobToken) -> Vec<(FileId, FileSymbol)> {
        self.imp.approximately_resolve_symbol(file_id, offset, token)
    }
    pub fn parent_module(&self, file_id: FileId) -> Vec<(FileId, FileSymbol)> {
        self.imp.parent_module(file_id)
    }
    pub fn crate_for(&self, file_id: FileId) -> Vec<CrateId> {
        self.imp.crate_for(file_id)
    }
    pub fn crate_root(&self, crate_id: CrateId) -> FileId {
        self.imp.crate_root(crate_id)
    }
    pub fn runnables(&self, file_id: FileId) -> Vec<Runnable> {
        let file = self.imp.file_syntax(file_id);
        libeditor::runnables(file)
    }
    pub fn highlight(&self, file_id: FileId) -> Vec<HighlightedRange> {
        let file = self.imp.file_syntax(file_id);
        libeditor::highlight(file)
    }
    pub fn completions(&self, file_id: FileId, offset: TextUnit) -> Option<Vec<CompletionItem>> {
        let file = self.imp.file_syntax(file_id);
        libeditor::scope_completion(file, offset)
    }
    pub fn assists(&self, file_id: FileId, range: TextRange) -> Vec<SourceChange> {
        self.imp.assists(file_id, range)
    }
    pub fn diagnostics(&self, file_id: FileId) -> Vec<Diagnostic> {
        self.imp.diagnostics(file_id)
    }
}

#[derive(Debug)]
pub struct LibraryData {
    root: roots::ReadonlySourceRoot
}

impl LibraryData {
    pub fn prepare(files: Vec<(FileId, String)>, file_resolver: Arc<FileResolver>) -> LibraryData {
        let file_resolver = FileResolverImp::new(file_resolver);
        let root = roots::ReadonlySourceRoot::new(files, file_resolver);
        LibraryData { root }
    }
}
