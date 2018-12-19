//! ra_analyzer crate is the brain of Rust analyzer. It relies on the `salsa`
//! crate, which provides and incremental on-demand database of facts.

macro_rules! ctry {
    ($expr:expr) => {
        match $expr {
            None => return Ok(None),
            Some(it) => it,
        }
    };
}

mod db;
mod imp;
mod completion;
mod symbol_index;
pub mod mock_analysis;

use std::{fmt, sync::Arc};

use rustc_hash::FxHashMap;
use ra_syntax::{SourceFileNode, TextRange, TextUnit};
use ra_text_edit::AtomTextEdit;
use rayon::prelude::*;
use relative_path::RelativePathBuf;

use crate::{
    imp::{AnalysisHostImpl, AnalysisImpl},
    symbol_index::SymbolIndex,
};

pub use crate::{
    completion::CompletionItem,
};
pub use ra_editor::{
    FileSymbol, Fold, FoldKind, HighlightedRange, LineIndex, Runnable, RunnableKind, StructureNode,
};
pub use hir::FnSignatureInfo;

pub use ra_db::{
    Canceled, Cancelable, FilePosition,
    CrateGraph, CrateId, SourceRootId, FileId, WORKSPACE
};

#[derive(Default)]
pub struct AnalysisChange {
    new_roots: Vec<SourceRootId>,
    roots_changed: FxHashMap<SourceRootId, RootChange>,
    files_changed: Vec<(FileId, Arc<String>)>,
    libraries_added: Vec<LibraryData>,
    crate_graph: Option<CrateGraph>,
}

#[derive(Default)]
struct RootChange {
    added: Vec<AddFile>,
    removed: Vec<RemoveFile>,
}

#[derive(Debug)]
struct AddFile {
    file_id: FileId,
    path: RelativePathBuf,
    text: Arc<String>,
}

#[derive(Debug)]
struct RemoveFile {
    file_id: FileId,
    path: RelativePathBuf,
}

impl fmt::Debug for AnalysisChange {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("AnalysisChange")
            .field("new_roots", &self.new_roots)
            .field("roots_changed", &self.roots_changed)
            .field("files_changed", &self.files_changed.len())
            .field("libraries_added", &self.libraries_added.len())
            .field("crate_graph", &self.crate_graph)
            .finish()
    }
}

impl fmt::Debug for RootChange {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("AnalysisChange")
            .field("added", &self.added.len())
            .field("removed", &self.removed.len())
            .finish()
    }
}

impl AnalysisChange {
    pub fn new() -> AnalysisChange {
        AnalysisChange::default()
    }
    pub fn add_root(&mut self, root_id: SourceRootId) {
        self.new_roots.push(root_id);
    }
    pub fn add_file(
        &mut self,
        root_id: SourceRootId,
        file_id: FileId,
        path: RelativePathBuf,
        text: Arc<String>,
    ) {
        let file = AddFile {
            file_id,
            path,
            text,
        };
        self.roots_changed
            .entry(root_id)
            .or_default()
            .added
            .push(file);
    }
    pub fn change_file(&mut self, file_id: FileId, new_text: Arc<String>) {
        self.files_changed.push((file_id, new_text))
    }
    pub fn remove_file(&mut self, root_id: SourceRootId, file_id: FileId, path: RelativePathBuf) {
        let file = RemoveFile { file_id, path };
        self.roots_changed
            .entry(root_id)
            .or_default()
            .removed
            .push(file);
    }
    pub fn add_library(&mut self, data: LibraryData) {
        self.libraries_added.push(data)
    }
    pub fn set_crate_graph(&mut self, graph: CrateGraph) {
        self.crate_graph = Some(graph);
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
    pub edits: Vec<AtomTextEdit>,
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

/// Result of "goto def" query.
#[derive(Debug)]
pub struct ReferenceResolution {
    /// The range of the reference itself. Client does not know what constitutes
    /// a reference, it handles us only the offset. It's helpful to tell the
    /// client where the reference was.
    pub reference_range: TextRange,
    /// What this reference resolves to.
    pub resolves_to: Vec<(FileId, FileSymbol)>,
}

impl ReferenceResolution {
    fn new(reference_range: TextRange) -> ReferenceResolution {
        ReferenceResolution {
            reference_range,
            resolves_to: Vec::new(),
        }
    }

    fn add_resolution(&mut self, file_id: FileId, symbol: FileSymbol) {
        self.resolves_to.push((file_id, symbol))
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
    ) -> Cancelable<Option<ReferenceResolution>> {
        self.imp.approximately_resolve_symbol(position)
    }
    pub fn find_all_refs(&self, position: FilePosition) -> Cancelable<Vec<(FileId, TextRange)>> {
        self.imp.find_all_refs(position)
    }
    pub fn doc_comment_for(
        &self,
        file_id: FileId,
        symbol: FileSymbol,
    ) -> Cancelable<Option<String>> {
        self.imp.doc_comment_for(file_id, symbol)
    }
    pub fn doc_text_for(&self, file_id: FileId, symbol: FileSymbol) -> Cancelable<Option<String>> {
        self.imp.doc_text_for(file_id, symbol)
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
    ) -> Cancelable<Option<(FnSignatureInfo, Option<usize>)>> {
        self.imp.resolve_callable(position)
    }
}

#[derive(Debug)]
pub struct LibraryData {
    root_id: SourceRootId,
    root_change: RootChange,
    symbol_index: SymbolIndex,
}

impl LibraryData {
    pub fn prepare(
        root_id: SourceRootId,
        files: Vec<(FileId, RelativePathBuf, Arc<String>)>,
    ) -> LibraryData {
        let symbol_index = SymbolIndex::for_files(files.par_iter().map(|(file_id, _, text)| {
            let file = SourceFileNode::parse(text);
            (*file_id, file)
        }));
        let mut root_change = RootChange::default();
        root_change.added = files
            .into_iter()
            .map(|(file_id, path, text)| AddFile {
                file_id,
                path,
                text,
            })
            .collect();
        LibraryData {
            root_id,
            root_change,
            symbol_index,
        }
    }
}

#[test]
fn analysis_is_send() {
    fn is_send<T: Send>() {}
    is_send::<Analysis>();
}
