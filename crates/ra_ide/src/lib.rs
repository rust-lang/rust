//! ra_ide crate provides "ide-centric" APIs for the rust-analyzer. That is,
//! it generally operates with files and text ranges, and returns results as
//! Strings, suitable for displaying to the human.
//!
//! What powers this API are the `RootDatabase` struct, which defines a `salsa`
//! database, and the `ra_hir` crate, where majority of the analysis happens.
//! However, IDE specific bits of the analysis (most notably completion) happen
//! in this crate.

// For proving that RootDatabase is RefUnwindSafe.
#![recursion_limit = "128"]

mod db;
pub mod mock_analysis;
mod symbol_index;
mod change;
mod source_change;
mod feature_flags;

mod status;
mod completion;
mod runnables;
mod goto_definition;
mod goto_type_definition;
mod extend_selection;
mod hover;
mod call_hierarchy;
mod call_info;
mod syntax_highlighting;
mod parent_module;
mod references;
mod impls;
mod assists;
mod diagnostics;
mod syntax_tree;
mod folding_ranges;
mod line_index;
mod line_index_utils;
mod join_lines;
mod typing;
mod matching_brace;
mod display;
mod inlay_hints;
mod wasm_shims;
mod expand;
mod expand_macro;

#[cfg(test)]
mod marks;
#[cfg(test)]
mod test_utils;

use std::sync::Arc;

use ra_cfg::CfgOptions;
use ra_db::{
    salsa::{self, ParallelDatabase},
    CheckCanceled, Env, FileLoader, SourceDatabase,
};
use ra_syntax::{SourceFile, TextRange, TextUnit};

use crate::{db::LineIndexDatabase, display::ToNav, symbol_index::FileSymbol};

pub use crate::{
    assists::{Assist, AssistId},
    call_hierarchy::CallItem,
    change::{AnalysisChange, LibraryData},
    completion::{CompletionItem, CompletionItemKind, InsertTextFormat},
    diagnostics::Severity,
    display::{file_structure, FunctionSignature, NavigationTarget, StructureNode},
    expand_macro::ExpandedMacro,
    feature_flags::FeatureFlags,
    folding_ranges::{Fold, FoldKind},
    hover::HoverResult,
    inlay_hints::{InlayHint, InlayKind},
    line_index::{LineCol, LineIndex},
    line_index_utils::translate_offset_with_edit,
    references::{Reference, ReferenceKind, ReferenceSearchResult, SearchScope},
    runnables::{Runnable, RunnableKind},
    source_change::{FileSystemEdit, SourceChange, SourceFileEdit},
    syntax_highlighting::HighlightedRange,
};

pub use hir::Documentation;
pub use ra_db::{
    Canceled, CrateGraph, CrateId, Edition, FileId, FilePosition, FileRange, SourceRootId,
};

pub type Cancelable<T> = Result<T, Canceled>;

#[derive(Debug)]
pub struct Diagnostic {
    pub message: String,
    pub range: TextRange,
    pub fix: Option<SourceChange>,
    pub severity: Severity,
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

/// Info associated with a text range.
#[derive(Debug)]
pub struct RangeInfo<T> {
    pub range: TextRange,
    pub info: T,
}

impl<T> RangeInfo<T> {
    pub fn new(range: TextRange, info: T) -> RangeInfo<T> {
        RangeInfo { range, info }
    }
}

/// Contains information about a call site. Specifically the
/// `FunctionSignature`and current parameter.
#[derive(Debug)]
pub struct CallInfo {
    pub signature: FunctionSignature,
    pub active_parameter: Option<usize>,
}

/// `AnalysisHost` stores the current state of the world.
#[derive(Debug)]
pub struct AnalysisHost {
    db: db::RootDatabase,
}

impl Default for AnalysisHost {
    fn default() -> AnalysisHost {
        AnalysisHost::new(None, FeatureFlags::default())
    }
}

impl AnalysisHost {
    pub fn new(lru_capcity: Option<usize>, feature_flags: FeatureFlags) -> AnalysisHost {
        AnalysisHost { db: db::RootDatabase::new(lru_capcity, feature_flags) }
    }
    /// Returns a snapshot of the current state, which you can query for
    /// semantic information.
    pub fn analysis(&self) -> Analysis {
        Analysis { db: self.db.snapshot() }
    }

    pub fn feature_flags(&self) -> &FeatureFlags {
        &self.db.feature_flags
    }

    /// Applies changes to the current state of the world. If there are
    /// outstanding snapshots, they will be canceled.
    pub fn apply_change(&mut self, change: AnalysisChange) {
        self.db.apply_change(change)
    }

    pub fn maybe_collect_garbage(&mut self) {
        self.db.maybe_collect_garbage();
    }

    pub fn collect_garbage(&mut self) {
        self.db.collect_garbage();
    }
    /// NB: this clears the database
    pub fn per_query_memory_usage(&mut self) -> Vec<(String, ra_prof::Bytes)> {
        self.db.per_query_memory_usage()
    }
    pub fn raw_database(
        &self,
    ) -> &(impl hir::db::HirDatabase + salsa::Database + ra_db::SourceDatabaseExt) {
        &self.db
    }
    pub fn raw_database_mut(
        &mut self,
    ) -> &mut (impl hir::db::HirDatabase + salsa::Database + ra_db::SourceDatabaseExt) {
        &mut self.db
    }
}

/// Analysis is a snapshot of a world state at a moment in time. It is the main
/// entry point for asking semantic information about the world. When the world
/// state is advanced using `AnalysisHost::apply_change` method, all existing
/// `Analysis` are canceled (most method return `Err(Canceled)`).
#[derive(Debug)]
pub struct Analysis {
    db: salsa::Snapshot<db::RootDatabase>,
}

// As a general design guideline, `Analysis` API are intended to be independent
// from the language server protocol. That is, when exposing some functionality
// we should think in terms of "what API makes most sense" and not in terms of
// "what types LSP uses". Although currently LSP is the only consumer of the
// API, the API should in theory be usable as a library, or via a different
// protocol.
impl Analysis {
    // Creates an analysis instance for a single file, without any extenal
    // dependencies, stdlib support or ability to apply changes. See
    // `AnalysisHost` for creating a fully-featured analysis.
    pub fn from_single_file(text: String) -> (Analysis, FileId) {
        let mut host = AnalysisHost::default();
        let source_root = SourceRootId(0);
        let mut change = AnalysisChange::new();
        change.add_root(source_root, true);
        let mut crate_graph = CrateGraph::default();
        let file_id = FileId(0);
        // FIXME: cfg options
        // Default to enable test for single file.
        let mut cfg_options = CfgOptions::default();
        cfg_options.insert_atom("test".into());
        crate_graph.add_crate_root(file_id, Edition::Edition2018, cfg_options, Env::default());
        change.add_file(source_root, file_id, "main.rs".into(), Arc::new(text));
        change.set_crate_graph(crate_graph);
        host.apply_change(change);
        (host.analysis(), file_id)
    }

    /// Features for Analysis.
    pub fn feature_flags(&self) -> &FeatureFlags {
        &self.db.feature_flags
    }

    /// Debug info about the current state of the analysis.
    pub fn status(&self) -> Cancelable<String> {
        self.with_db(|db| status::status(&*db))
    }

    /// Gets the text of the source file.
    pub fn file_text(&self, file_id: FileId) -> Cancelable<Arc<String>> {
        self.with_db(|db| db.file_text(file_id))
    }

    /// Gets the syntax tree of the file.
    pub fn parse(&self, file_id: FileId) -> Cancelable<SourceFile> {
        self.with_db(|db| db.parse(file_id).tree())
    }

    /// Gets the file's `LineIndex`: data structure to convert between absolute
    /// offsets and line/column representation.
    pub fn file_line_index(&self, file_id: FileId) -> Cancelable<Arc<LineIndex>> {
        self.with_db(|db| db.line_index(file_id))
    }

    /// Selects the next syntactic nodes encompassing the range.
    pub fn extend_selection(&self, frange: FileRange) -> Cancelable<TextRange> {
        self.with_db(|db| extend_selection::extend_selection(db, frange))
    }

    /// Returns position of the matching brace (all types of braces are
    /// supported).
    pub fn matching_brace(&self, position: FilePosition) -> Cancelable<Option<TextUnit>> {
        self.with_db(|db| {
            let parse = db.parse(position.file_id);
            let file = parse.tree();
            matching_brace::matching_brace(&file, position.offset)
        })
    }

    /// Returns a syntax tree represented as `String`, for debug purposes.
    // FIXME: use a better name here.
    pub fn syntax_tree(
        &self,
        file_id: FileId,
        text_range: Option<TextRange>,
    ) -> Cancelable<String> {
        self.with_db(|db| syntax_tree::syntax_tree(&db, file_id, text_range))
    }

    pub fn expand_macro(&self, position: FilePosition) -> Cancelable<Option<ExpandedMacro>> {
        self.with_db(|db| expand_macro::expand_macro(db, position))
    }

    /// Returns an edit to remove all newlines in the range, cleaning up minor
    /// stuff like trailing commas.
    pub fn join_lines(&self, frange: FileRange) -> Cancelable<SourceChange> {
        self.with_db(|db| {
            let parse = db.parse(frange.file_id);
            let file_edit = SourceFileEdit {
                file_id: frange.file_id,
                edit: join_lines::join_lines(&parse.tree(), frange.range),
            };
            SourceChange::source_file_edit("join lines", file_edit)
        })
    }

    /// Returns an edit which should be applied when opening a new line, fixing
    /// up minor stuff like continuing the comment.
    pub fn on_enter(&self, position: FilePosition) -> Cancelable<Option<SourceChange>> {
        self.with_db(|db| typing::on_enter(&db, position))
    }

    /// Returns an edit which should be applied after a character was typed.
    ///
    /// This is useful for some on-the-fly fixups, like adding `;` to `let =`
    /// automatically.
    pub fn on_char_typed(
        &self,
        position: FilePosition,
        char_typed: char,
    ) -> Cancelable<Option<SourceChange>> {
        // Fast path to not even parse the file.
        if !typing::TRIGGER_CHARS.contains(char_typed) {
            return Ok(None);
        }
        self.with_db(|db| typing::on_char_typed(&db, position, char_typed))
    }

    /// Returns a tree representation of symbols in the file. Useful to draw a
    /// file outline.
    pub fn file_structure(&self, file_id: FileId) -> Cancelable<Vec<StructureNode>> {
        self.with_db(|db| file_structure(&db.parse(file_id).tree()))
    }

    /// Returns a list of the places in the file where type hints can be displayed.
    pub fn inlay_hints(
        &self,
        file_id: FileId,
        max_inlay_hint_length: Option<usize>,
    ) -> Cancelable<Vec<InlayHint>> {
        self.with_db(|db| {
            inlay_hints::inlay_hints(db, file_id, &db.parse(file_id).tree(), max_inlay_hint_length)
        })
    }

    /// Returns the set of folding ranges.
    pub fn folding_ranges(&self, file_id: FileId) -> Cancelable<Vec<Fold>> {
        self.with_db(|db| folding_ranges::folding_ranges(&db.parse(file_id).tree()))
    }

    /// Fuzzy searches for a symbol.
    pub fn symbol_search(&self, query: Query) -> Cancelable<Vec<NavigationTarget>> {
        self.with_db(|db| {
            symbol_index::world_symbols(db, query)
                .into_iter()
                .map(|s| s.to_nav(db))
                .collect::<Vec<_>>()
        })
    }

    /// Returns the definitions from the symbol at `position`.
    pub fn goto_definition(
        &self,
        position: FilePosition,
    ) -> Cancelable<Option<RangeInfo<Vec<NavigationTarget>>>> {
        self.with_db(|db| goto_definition::goto_definition(db, position))
    }

    /// Returns the impls from the symbol at `position`.
    pub fn goto_implementation(
        &self,
        position: FilePosition,
    ) -> Cancelable<Option<RangeInfo<Vec<NavigationTarget>>>> {
        self.with_db(|db| impls::goto_implementation(db, position))
    }

    /// Returns the type definitions for the symbol at `position`.
    pub fn goto_type_definition(
        &self,
        position: FilePosition,
    ) -> Cancelable<Option<RangeInfo<Vec<NavigationTarget>>>> {
        self.with_db(|db| goto_type_definition::goto_type_definition(db, position))
    }

    /// Finds all usages of the reference at point.
    pub fn find_all_refs(
        &self,
        position: FilePosition,
        search_scope: Option<SearchScope>,
    ) -> Cancelable<Option<ReferenceSearchResult>> {
        self.with_db(|db| references::find_all_refs(db, position, search_scope).map(|it| it.info))
    }

    /// Returns a short text describing element at position.
    pub fn hover(&self, position: FilePosition) -> Cancelable<Option<RangeInfo<HoverResult>>> {
        self.with_db(|db| hover::hover(db, position))
    }

    /// Computes parameter information for the given call expression.
    pub fn call_info(&self, position: FilePosition) -> Cancelable<Option<CallInfo>> {
        self.with_db(|db| call_info::call_info(db, position))
    }

    /// Computes call hierarchy candidates for the given file position.
    pub fn call_hierarchy(
        &self,
        position: FilePosition,
    ) -> Cancelable<Option<RangeInfo<Vec<NavigationTarget>>>> {
        self.with_db(|db| call_hierarchy::call_hierarchy(db, position))
    }

    /// Computes incoming calls for the given file position.
    pub fn incoming_calls(&self, position: FilePosition) -> Cancelable<Option<Vec<CallItem>>> {
        self.with_db(|db| call_hierarchy::incoming_calls(db, position))
    }

    /// Computes incoming calls for the given file position.
    pub fn outgoing_calls(&self, position: FilePosition) -> Cancelable<Option<Vec<CallItem>>> {
        self.with_db(|db| call_hierarchy::outgoing_calls(db, position))
    }

    /// Returns a `mod name;` declaration which created the current module.
    pub fn parent_module(&self, position: FilePosition) -> Cancelable<Vec<NavigationTarget>> {
        self.with_db(|db| parent_module::parent_module(db, position))
    }

    /// Returns crates this file belongs too.
    pub fn crate_for(&self, file_id: FileId) -> Cancelable<Vec<CrateId>> {
        self.with_db(|db| parent_module::crate_for(db, file_id))
    }

    /// Returns the edition of the given crate.
    pub fn crate_edition(&self, crate_id: CrateId) -> Cancelable<Edition> {
        self.with_db(|db| db.crate_graph().edition(crate_id))
    }

    /// Returns the root file of the given crate.
    pub fn crate_root(&self, crate_id: CrateId) -> Cancelable<FileId> {
        self.with_db(|db| db.crate_graph().crate_root(crate_id))
    }

    /// Returns the set of possible targets to run for the current file.
    pub fn runnables(&self, file_id: FileId) -> Cancelable<Vec<Runnable>> {
        self.with_db(|db| runnables::runnables(db, file_id))
    }

    /// Computes syntax highlighting for the given file.
    pub fn highlight(&self, file_id: FileId) -> Cancelable<Vec<HighlightedRange>> {
        self.with_db(|db| syntax_highlighting::highlight(db, file_id))
    }

    /// Computes syntax highlighting for the given file.
    pub fn highlight_as_html(&self, file_id: FileId, rainbow: bool) -> Cancelable<String> {
        self.with_db(|db| syntax_highlighting::highlight_as_html(db, file_id, rainbow))
    }

    /// Computes completions at the given position.
    pub fn completions(&self, position: FilePosition) -> Cancelable<Option<Vec<CompletionItem>>> {
        self.with_db(|db| completion::completions(db, position).map(Into::into))
    }

    /// Computes assists (aka code actions aka intentions) for the given
    /// position.
    pub fn assists(&self, frange: FileRange) -> Cancelable<Vec<Assist>> {
        self.with_db(|db| assists::assists(db, frange))
    }

    /// Computes the set of diagnostics for the given file.
    pub fn diagnostics(&self, file_id: FileId) -> Cancelable<Vec<Diagnostic>> {
        self.with_db(|db| diagnostics::diagnostics(db, file_id))
    }

    /// Computes the type of the expression at the given position.
    pub fn type_of(&self, frange: FileRange) -> Cancelable<Option<String>> {
        self.with_db(|db| hover::type_of(db, frange))
    }

    /// Returns the edit required to rename reference at the position to the new
    /// name.
    pub fn rename(
        &self,
        position: FilePosition,
        new_name: &str,
    ) -> Cancelable<Option<RangeInfo<SourceChange>>> {
        self.with_db(|db| references::rename(db, position, new_name))
    }

    /// Performs an operation on that may be Canceled.
    fn with_db<F: FnOnce(&db::RootDatabase) -> T + std::panic::UnwindSafe, T>(
        &self,
        f: F,
    ) -> Cancelable<T> {
        self.db.catch_canceled(f)
    }
}

#[test]
fn analysis_is_send() {
    fn is_send<T: Send>() {}
    is_send::<Analysis>();
}
