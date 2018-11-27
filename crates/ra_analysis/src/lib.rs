//! ra_analyzer crate is the brain of Rust analyzer. It relies on the `salsa`
//! crate, which provides and incremental on-demand database of facts.

extern crate fst;
extern crate ra_editor;
extern crate ra_syntax;
extern crate rayon;
extern crate relative_path;
extern crate rustc_hash;
extern crate salsa;

macro_rules! ctry {
    ($expr:expr) => {
        match $expr {
            None => return Ok(None),
            Some(it) => it,
        }
    };
}

mod arena;
mod syntax_ptr;
mod input;
mod db;
mod loc2id;
mod imp;
mod completion;
mod hir;
mod symbol_index;
pub mod mock_analysis;

use std::{fmt, sync::Arc};

use ra_syntax::{AtomEdit, SourceFileNode, TextRange, TextUnit};
use rayon::prelude::*;
use relative_path::RelativePathBuf;

use crate::{
    imp::{AnalysisHostImpl, AnalysisImpl, FileResolverImp},
    symbol_index::SymbolIndex,
};

pub use crate::{
    completion::CompletionItem,
    hir::FnDescriptor,
    input::{CrateGraph, CrateId, FileId, FileResolver},
};
pub use ra_editor::{
    FileSymbol, Fold, FoldKind, HighlightedRange, LineIndex, Runnable, RunnableKind, StructureNode,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Canceled;

pub type Cancelable<T> = Result<T, Canceled>;

impl std::fmt::Display for Canceled {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_str("Canceled")
    }
}

impl std::error::Error for Canceled {}

#[derive(Default)]
pub struct AnalysisChange {
    files_added: Vec<(FileId, String)>,
    files_changed: Vec<(FileId, String)>,
    files_removed: Vec<(FileId)>,
    libraries_added: Vec<LibraryData>,
    crate_graph: Option<CrateGraph>,
    file_resolver: Option<FileResolverImp>,
}

impl fmt::Debug for AnalysisChange {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("AnalysisChange")
            .field("files_added", &self.files_added.len())
            .field("files_changed", &self.files_changed.len())
            .field("files_removed", &self.files_removed.len())
            .field("libraries_added", &self.libraries_added.len())
            .field("crate_graph", &self.crate_graph)
            .field("file_resolver", &self.file_resolver)
            .finish()
    }
}

impl AnalysisChange {
    pub fn new() -> AnalysisChange {
        AnalysisChange::default()
    }
    pub fn add_file(&mut self, file_id: FileId, text: String) {
        self.files_added.push((file_id, text))
    }
    pub fn change_file(&mut self, file_id: FileId, new_text: String) {
        self.files_changed.push((file_id, new_text))
    }
    pub fn remove_file(&mut self, file_id: FileId) {
        self.files_removed.push(file_id)
    }
    pub fn add_library(&mut self, data: LibraryData) {
        self.libraries_added.push(data)
    }
    pub fn set_crate_graph(&mut self, graph: CrateGraph) {
        self.crate_graph = Some(graph);
    }
    pub fn set_file_resolver(&mut self, file_resolver: Arc<FileResolver>) {
        self.file_resolver = Some(FileResolverImp::new(file_resolver));
    }
}

/// `AnalysisHost` stores the current state of the world.
#[derive(Debug, Default)]
pub struct AnalysisHost {
    imp: AnalysisHostImpl,
}

impl AnalysisHost {
    /// Returns a snapshot of the current state, which you can query for
    /// semantic information.
    pub fn analysis(&self) -> Analysis {
        Analysis {
            imp: self.imp.analysis(),
        }
    }
    /// Applies changes to the current state of the world. If there are
    /// outstanding snapshots, they will be canceled.
    pub fn apply_change(&mut self, change: AnalysisChange) {
        self.imp.apply_change(change)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FilePosition {
    pub file_id: FileId,
    pub offset: TextUnit,
}

#[derive(Debug)]
pub struct SourceChange {
    pub label: String,
    pub source_file_edits: Vec<SourceFileNodeEdit>,
    pub file_system_edits: Vec<FileSystemEdit>,
    pub cursor_position: Option<FilePosition>,
}

#[derive(Debug)]
pub struct SourceFileNodeEdit {
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
    },
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
            limit: usize::max_value(),
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

/// Analysis is a snapshot of a world state at a moment in time. It is the main
/// entry point for asking semantic information about the world. When the world
/// state is advanced using `AnalysisHost::apply_change` method, all existing
/// `Analysis` are canceled (most method return `Err(Canceled)`).
#[derive(Debug)]
pub struct Analysis {
    pub(crate) imp: AnalysisImpl,
}

impl Analysis {
    pub fn file_syntax(&self, file_id: FileId) -> SourceFileNode {
        self.imp.file_syntax(file_id).clone()
    }
    pub fn file_line_index(&self, file_id: FileId) -> Arc<LineIndex> {
        self.imp.file_line_index(file_id)
    }
    pub fn extend_selection(&self, file: &SourceFileNode, range: TextRange) -> TextRange {
        ra_editor::extend_selection(file, range).unwrap_or(range)
    }
    pub fn matching_brace(&self, file: &SourceFileNode, offset: TextUnit) -> Option<TextUnit> {
        ra_editor::matching_brace(file, offset)
    }
    pub fn syntax_tree(&self, file_id: FileId) -> String {
        let file = self.imp.file_syntax(file_id);
        ra_editor::syntax_tree(&file)
    }
    pub fn join_lines(&self, file_id: FileId, range: TextRange) -> SourceChange {
        let file = self.imp.file_syntax(file_id);
        SourceChange::from_local_edit(file_id, "join lines", ra_editor::join_lines(&file, range))
    }
    pub fn on_enter(&self, position: FilePosition) -> Option<SourceChange> {
        let file = self.imp.file_syntax(position.file_id);
        let edit = ra_editor::on_enter(&file, position.offset)?;
        let res = SourceChange::from_local_edit(position.file_id, "on enter", edit);
        Some(res)
    }
    pub fn on_eq_typed(&self, position: FilePosition) -> Option<SourceChange> {
        let file = self.imp.file_syntax(position.file_id);
        Some(SourceChange::from_local_edit(
            position.file_id,
            "add semicolon",
            ra_editor::on_eq_typed(&file, position.offset)?,
        ))
    }
    pub fn file_structure(&self, file_id: FileId) -> Vec<StructureNode> {
        let file = self.imp.file_syntax(file_id);
        ra_editor::file_structure(&file)
    }
    pub fn folding_ranges(&self, file_id: FileId) -> Vec<Fold> {
        let file = self.imp.file_syntax(file_id);
        ra_editor::folding_ranges(&file)
    }
    pub fn symbol_search(&self, query: Query) -> Cancelable<Vec<(FileId, FileSymbol)>> {
        self.imp.world_symbols(query)
    }
    pub fn approximately_resolve_symbol(
        &self,
        position: FilePosition,
    ) -> Cancelable<Vec<(FileId, FileSymbol)>> {
        self.imp.approximately_resolve_symbol(position)
    }
    pub fn find_all_refs(&self, position: FilePosition) -> Cancelable<Vec<(FileId, TextRange)>> {
        Ok(self.imp.find_all_refs(position))
    }
    pub fn doc_comment_for(
        &self,
        file_id: FileId,
        symbol: FileSymbol,
    ) -> Cancelable<Option<String>> {
        self.imp.doc_comment_for(file_id, symbol)
    }
    pub fn parent_module(&self, position: FilePosition) -> Cancelable<Vec<(FileId, FileSymbol)>> {
        self.imp.parent_module(position)
    }
    pub fn crate_for(&self, file_id: FileId) -> Cancelable<Vec<CrateId>> {
        self.imp.crate_for(file_id)
    }
    pub fn crate_root(&self, crate_id: CrateId) -> Cancelable<FileId> {
        Ok(self.imp.crate_root(crate_id))
    }
    pub fn runnables(&self, file_id: FileId) -> Cancelable<Vec<Runnable>> {
        let file = self.imp.file_syntax(file_id);
        Ok(ra_editor::runnables(&file))
    }
    pub fn highlight(&self, file_id: FileId) -> Cancelable<Vec<HighlightedRange>> {
        let file = self.imp.file_syntax(file_id);
        Ok(ra_editor::highlight(&file))
    }
    pub fn completions(&self, position: FilePosition) -> Cancelable<Option<Vec<CompletionItem>>> {
        self.imp.completions(position)
    }
    pub fn assists(&self, file_id: FileId, range: TextRange) -> Cancelable<Vec<SourceChange>> {
        Ok(self.imp.assists(file_id, range))
    }
    pub fn diagnostics(&self, file_id: FileId) -> Cancelable<Vec<Diagnostic>> {
        self.imp.diagnostics(file_id)
    }
    pub fn resolve_callable(
        &self,
        position: FilePosition,
    ) -> Cancelable<Option<(FnDescriptor, Option<usize>)>> {
        self.imp.resolve_callable(position)
    }
}

#[derive(Debug)]
pub struct LibraryData {
    files: Vec<(FileId, String)>,
    file_resolver: FileResolverImp,
    symbol_index: SymbolIndex,
}

impl LibraryData {
    pub fn prepare(files: Vec<(FileId, String)>, file_resolver: Arc<FileResolver>) -> LibraryData {
        let symbol_index = SymbolIndex::for_files(files.par_iter().map(|(file_id, text)| {
            let file = SourceFileNode::parse(text);
            (*file_id, file)
        }));
        LibraryData {
            files,
            file_resolver: FileResolverImp::new(file_resolver),
            symbol_index,
        }
    }
}

#[test]
fn analysis_is_send() {
    fn is_send<T: Send>() {}
    is_send::<Analysis>();
}
