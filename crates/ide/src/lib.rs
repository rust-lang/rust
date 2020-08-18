//! ide crate provides "ide-centric" APIs for the rust-analyzer. That is,
//! it generally operates with files and text ranges, and returns results as
//! Strings, suitable for displaying to the human.
//!
//! What powers this API are the `RootDatabase` struct, which defines a `salsa`
//! database, and the `hir` crate, where majority of the analysis happens.
//! However, IDE specific bits of the analysis (most notably completion) happen
//! in this crate.

// For proving that RootDatabase is RefUnwindSafe.
#![recursion_limit = "128"]

#[allow(unused)]
macro_rules! eprintln {
    ($($tt:tt)*) => { stdx::eprintln!($($tt)*) };
}

pub mod mock_analysis;

mod markup;
mod prime_caches;
mod display;

mod call_hierarchy;
mod call_info;
mod completion;
mod diagnostics;
mod expand_macro;
mod extend_selection;
mod file_structure;
mod folding_ranges;
mod goto_definition;
mod goto_implementation;
mod goto_type_definition;
mod hover;
mod inlay_hints;
mod join_lines;
mod matching_brace;
mod parent_module;
mod references;
mod runnables;
mod status;
mod syntax_highlighting;
mod syntax_tree;
mod typing;

use std::sync::Arc;

use base_db::{
    salsa::{self, ParallelDatabase},
    CheckCanceled, Env, FileLoader, FileSet, SourceDatabase, VfsPath,
};
use cfg::CfgOptions;
use ide_db::{
    symbol_index::{self, FileSymbol},
    LineIndexDatabase,
};
use syntax::{SourceFile, TextRange, TextSize};

use crate::display::ToNav;

pub use crate::{
    call_hierarchy::CallItem,
    call_info::CallInfo,
    completion::{
        CompletionConfig, CompletionItem, CompletionItemKind, CompletionScore, InsertTextFormat,
    },
    diagnostics::{Diagnostic, DiagnosticsConfig, Fix, Severity},
    display::NavigationTarget,
    expand_macro::ExpandedMacro,
    file_structure::StructureNode,
    folding_ranges::{Fold, FoldKind},
    hover::{HoverAction, HoverConfig, HoverGotoTypeData, HoverResult},
    inlay_hints::{InlayHint, InlayHintsConfig, InlayKind},
    markup::Markup,
    references::{Declaration, Reference, ReferenceAccess, ReferenceKind, ReferenceSearchResult},
    runnables::{Runnable, RunnableKind, TestId},
    syntax_highlighting::{
        Highlight, HighlightModifier, HighlightModifiers, HighlightTag, HighlightedRange,
    },
};

pub use assists::{Assist, AssistConfig, AssistId, AssistKind, ResolvedAssist};
pub use base_db::{
    Canceled, CrateGraph, CrateId, Edition, FileId, FilePosition, FileRange, SourceRoot,
    SourceRootId,
};
pub use hir::{Documentation, Semantics};
pub use ide_db::{
    change::AnalysisChange,
    label::Label,
    line_index::{LineCol, LineIndex},
    search::SearchScope,
    source_change::{FileSystemEdit, SourceChange, SourceFileEdit},
    symbol_index::Query,
    RootDatabase,
};
pub use ssr::SsrError;
pub use text_edit::{Indel, TextEdit};

pub type Cancelable<T> = Result<T, Canceled>;

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

/// `AnalysisHost` stores the current state of the world.
#[derive(Debug)]
pub struct AnalysisHost {
    db: RootDatabase,
}

impl AnalysisHost {
    pub fn new(lru_capacity: Option<usize>) -> AnalysisHost {
        AnalysisHost { db: RootDatabase::new(lru_capacity) }
    }

    pub fn update_lru_capacity(&mut self, lru_capacity: Option<usize>) {
        self.db.update_lru_capacity(lru_capacity);
    }

    /// Returns a snapshot of the current state, which you can query for
    /// semantic information.
    pub fn analysis(&self) -> Analysis {
        Analysis { db: self.db.snapshot() }
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
    pub fn per_query_memory_usage(&mut self) -> Vec<(String, profile::Bytes)> {
        self.db.per_query_memory_usage()
    }
    pub fn request_cancellation(&mut self) {
        self.db.request_cancellation();
    }
    pub fn raw_database(&self) -> &RootDatabase {
        &self.db
    }
    pub fn raw_database_mut(&mut self) -> &mut RootDatabase {
        &mut self.db
    }
}

impl Default for AnalysisHost {
    fn default() -> AnalysisHost {
        AnalysisHost::new(None)
    }
}

/// Analysis is a snapshot of a world state at a moment in time. It is the main
/// entry point for asking semantic information about the world. When the world
/// state is advanced using `AnalysisHost::apply_change` method, all existing
/// `Analysis` are canceled (most method return `Err(Canceled)`).
#[derive(Debug)]
pub struct Analysis {
    db: salsa::Snapshot<RootDatabase>,
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
        let file_id = FileId(0);
        let mut file_set = FileSet::default();
        file_set.insert(file_id, VfsPath::new_virtual_path("/main.rs".to_string()));
        let source_root = SourceRoot::new_local(file_set);

        let mut change = AnalysisChange::new();
        change.set_roots(vec![source_root]);
        let mut crate_graph = CrateGraph::default();
        // FIXME: cfg options
        // Default to enable test for single file.
        let mut cfg_options = CfgOptions::default();
        cfg_options.insert_atom("test".into());
        crate_graph.add_crate_root(
            file_id,
            Edition::Edition2018,
            None,
            cfg_options,
            Env::default(),
            Default::default(),
        );
        change.change_file(file_id, Some(Arc::new(text)));
        change.set_crate_graph(crate_graph);
        host.apply_change(change);
        (host.analysis(), file_id)
    }

    /// Debug info about the current state of the analysis.
    pub fn status(&self) -> Cancelable<String> {
        self.with_db(|db| status::status(&*db))
    }

    pub fn prime_caches(&self, files: Vec<FileId>) -> Cancelable<()> {
        self.with_db(|db| prime_caches::prime_caches(db, files))
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
    pub fn matching_brace(&self, position: FilePosition) -> Cancelable<Option<TextSize>> {
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
    pub fn join_lines(&self, frange: FileRange) -> Cancelable<TextEdit> {
        self.with_db(|db| {
            let parse = db.parse(frange.file_id);
            join_lines::join_lines(&parse.tree(), frange.range)
        })
    }

    /// Returns an edit which should be applied when opening a new line, fixing
    /// up minor stuff like continuing the comment.
    /// The edit will be a snippet (with `$0`).
    pub fn on_enter(&self, position: FilePosition) -> Cancelable<Option<TextEdit>> {
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
        self.with_db(|db| file_structure::file_structure(&db.parse(file_id).tree()))
    }

    /// Returns a list of the places in the file where type hints can be displayed.
    pub fn inlay_hints(
        &self,
        file_id: FileId,
        config: &InlayHintsConfig,
    ) -> Cancelable<Vec<InlayHint>> {
        self.with_db(|db| inlay_hints::inlay_hints(db, file_id, config))
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
        self.with_db(|db| goto_implementation::goto_implementation(db, position))
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
        self.with_db(|db| {
            references::find_all_refs(&Semantics::new(db), position, search_scope).map(|it| it.info)
        })
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
        self.with_db(|db| db.crate_graph()[crate_id].edition)
    }

    /// Returns the root file of the given crate.
    pub fn crate_root(&self, crate_id: CrateId) -> Cancelable<FileId> {
        self.with_db(|db| db.crate_graph()[crate_id].root_file_id)
    }

    /// Returns the set of possible targets to run for the current file.
    pub fn runnables(&self, file_id: FileId) -> Cancelable<Vec<Runnable>> {
        self.with_db(|db| runnables::runnables(db, file_id))
    }

    /// Computes syntax highlighting for the given file
    pub fn highlight(&self, file_id: FileId) -> Cancelable<Vec<HighlightedRange>> {
        self.with_db(|db| syntax_highlighting::highlight(db, file_id, None, false))
    }

    /// Computes syntax highlighting for the given file range.
    pub fn highlight_range(&self, frange: FileRange) -> Cancelable<Vec<HighlightedRange>> {
        self.with_db(|db| {
            syntax_highlighting::highlight(db, frange.file_id, Some(frange.range), false)
        })
    }

    /// Computes syntax highlighting for the given file.
    pub fn highlight_as_html(&self, file_id: FileId, rainbow: bool) -> Cancelable<String> {
        self.with_db(|db| syntax_highlighting::highlight_as_html(db, file_id, rainbow))
    }

    /// Computes completions at the given position.
    pub fn completions(
        &self,
        config: &CompletionConfig,
        position: FilePosition,
    ) -> Cancelable<Option<Vec<CompletionItem>>> {
        self.with_db(|db| completion::completions(db, config, position).map(Into::into))
    }

    /// Computes resolved assists with source changes for the given position.
    pub fn resolved_assists(
        &self,
        config: &AssistConfig,
        frange: FileRange,
    ) -> Cancelable<Vec<ResolvedAssist>> {
        self.with_db(|db| assists::Assist::resolved(db, config, frange))
    }

    /// Computes unresolved assists (aka code actions aka intentions) for the given
    /// position.
    pub fn unresolved_assists(
        &self,
        config: &AssistConfig,
        frange: FileRange,
    ) -> Cancelable<Vec<Assist>> {
        self.with_db(|db| Assist::unresolved(db, config, frange))
    }

    /// Computes the set of diagnostics for the given file.
    pub fn diagnostics(
        &self,
        config: &DiagnosticsConfig,
        file_id: FileId,
    ) -> Cancelable<Vec<Diagnostic>> {
        self.with_db(|db| diagnostics::diagnostics(db, config, file_id))
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

    pub fn structural_search_replace(
        &self,
        query: &str,
        parse_only: bool,
        resolve_context: FilePosition,
        selections: Vec<FileRange>,
    ) -> Cancelable<Result<SourceChange, SsrError>> {
        self.with_db(|db| {
            let rule: ssr::SsrRule = query.parse()?;
            let mut match_finder = ssr::MatchFinder::in_context(db, resolve_context, selections);
            match_finder.add_rule(rule)?;
            let edits = if parse_only { Vec::new() } else { match_finder.edits() };
            Ok(SourceChange::from(edits))
        })
    }

    /// Performs an operation on that may be Canceled.
    fn with_db<F, T>(&self, f: F) -> Cancelable<T>
    where
        F: FnOnce(&RootDatabase) -> T + std::panic::UnwindSafe,
    {
        self.db.catch_canceled(f)
    }
}

#[test]
fn analysis_is_send() {
    fn is_send<T: Send>() {}
    is_send::<Analysis>();
}
