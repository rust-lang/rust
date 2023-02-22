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
#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

#[allow(unused)]
macro_rules! eprintln {
    ($($tt:tt)*) => { stdx::eprintln!($($tt)*) };
}

#[cfg(test)]
mod fixture;

mod markup;
mod prime_caches;
mod navigation_target;

mod annotations;
mod call_hierarchy;
mod signature_help;
mod doc_links;
mod highlight_related;
mod expand_macro;
mod extend_selection;
mod file_structure;
mod folding_ranges;
mod goto_declaration;
mod goto_definition;
mod goto_implementation;
mod goto_type_definition;
mod hover;
mod inlay_hints;
mod join_lines;
mod markdown_remove;
mod matching_brace;
mod moniker;
mod move_item;
mod parent_module;
mod references;
mod rename;
mod runnables;
mod ssr;
mod static_index;
mod status;
mod syntax_highlighting;
mod syntax_tree;
mod typing;
mod view_crate_graph;
mod view_hir;
mod view_item_tree;
mod shuffle_crate_graph;

use std::sync::Arc;

use cfg::CfgOptions;
use ide_db::{
    base_db::{
        salsa::{self, ParallelDatabase},
        CrateOrigin, Env, FileLoader, FileSet, SourceDatabase, VfsPath,
    },
    symbol_index, LineIndexDatabase,
};
use syntax::SourceFile;

use crate::navigation_target::{ToNav, TryToNav};

pub use crate::{
    annotations::{Annotation, AnnotationConfig, AnnotationKind, AnnotationLocation},
    call_hierarchy::CallItem,
    expand_macro::ExpandedMacro,
    file_structure::{StructureNode, StructureNodeKind},
    folding_ranges::{Fold, FoldKind},
    highlight_related::{HighlightRelatedConfig, HighlightedRange},
    hover::{HoverAction, HoverConfig, HoverDocFormat, HoverGotoTypeData, HoverResult},
    inlay_hints::{
        AdjustmentHints, AdjustmentHintsMode, ClosureReturnTypeHints, DiscriminantHints, InlayHint,
        InlayHintLabel, InlayHintLabelPart, InlayHintsConfig, InlayKind, InlayTooltip,
        LifetimeElisionHints,
    },
    join_lines::JoinLinesConfig,
    markup::Markup,
    moniker::{MonikerDescriptorKind, MonikerKind, MonikerResult, PackageInformation},
    move_item::Direction,
    navigation_target::NavigationTarget,
    prime_caches::ParallelPrimeCachesProgress,
    references::ReferenceSearchResult,
    rename::RenameError,
    runnables::{Runnable, RunnableKind, TestId},
    signature_help::SignatureHelp,
    static_index::{StaticIndex, StaticIndexedFile, TokenId, TokenStaticData},
    syntax_highlighting::{
        tags::{Highlight, HlMod, HlMods, HlOperator, HlPunct, HlTag},
        HighlightConfig, HlRange,
    },
};
pub use hir::{Documentation, Semantics};
pub use ide_assists::{
    Assist, AssistConfig, AssistId, AssistKind, AssistResolveStrategy, SingleResolve,
};
pub use ide_completion::{
    CallableSnippets, CompletionConfig, CompletionItem, CompletionItemKind, CompletionRelevance,
    Snippet, SnippetScope,
};
pub use ide_db::{
    base_db::{
        Cancelled, Change, CrateGraph, CrateId, Edition, FileId, FilePosition, FileRange,
        SourceRoot, SourceRootId,
    },
    label::Label,
    line_index::{LineCol, LineIndex},
    search::{ReferenceCategory, SearchScope},
    source_change::{FileSystemEdit, SourceChange},
    symbol_index::Query,
    RootDatabase, SymbolKind,
};
pub use ide_diagnostics::{Diagnostic, DiagnosticsConfig, ExprFillDefaultMode, Severity};
pub use ide_ssr::SsrError;
pub use syntax::{TextRange, TextSize};
pub use text_edit::{Indel, TextEdit};

pub type Cancellable<T> = Result<T, Cancelled>;

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
    pub fn apply_change(&mut self, change: Change) {
        self.db.apply_change(change)
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

    pub fn shuffle_crate_graph(&mut self) {
        shuffle_crate_graph::shuffle_crate_graph(&mut self.db);
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
    // Creates an analysis instance for a single file, without any external
    // dependencies, stdlib support or ability to apply changes. See
    // `AnalysisHost` for creating a fully-featured analysis.
    pub fn from_single_file(text: String) -> (Analysis, FileId) {
        let mut host = AnalysisHost::default();
        let file_id = FileId(0);
        let mut file_set = FileSet::default();
        file_set.insert(file_id, VfsPath::new_virtual_path("/main.rs".to_string()));
        let source_root = SourceRoot::new_local(file_set);

        let mut change = Change::new();
        change.set_roots(vec![source_root]);
        let mut crate_graph = CrateGraph::default();
        // FIXME: cfg options
        // Default to enable test for single file.
        let mut cfg_options = CfgOptions::default();
        cfg_options.insert_atom("test".into());
        crate_graph.add_crate_root(
            file_id,
            Edition::CURRENT,
            None,
            None,
            cfg_options.clone(),
            cfg_options,
            Env::default(),
            Ok(Vec::new()),
            false,
            CrateOrigin::CratesIo { repo: None, name: None },
            Err("Analysis::from_single_file has no target layout".into()),
        );
        change.change_file(file_id, Some(Arc::new(text)));
        change.set_crate_graph(crate_graph);
        host.apply_change(change);
        (host.analysis(), file_id)
    }

    /// Debug info about the current state of the analysis.
    pub fn status(&self, file_id: Option<FileId>) -> Cancellable<String> {
        self.with_db(|db| status::status(&*db, file_id))
    }

    pub fn parallel_prime_caches<F>(&self, num_worker_threads: u8, cb: F) -> Cancellable<()>
    where
        F: Fn(ParallelPrimeCachesProgress) + Sync + std::panic::UnwindSafe,
    {
        self.with_db(move |db| prime_caches::parallel_prime_caches(db, num_worker_threads, &cb))
    }

    /// Gets the text of the source file.
    pub fn file_text(&self, file_id: FileId) -> Cancellable<Arc<String>> {
        self.with_db(|db| db.file_text(file_id))
    }

    /// Gets the syntax tree of the file.
    pub fn parse(&self, file_id: FileId) -> Cancellable<SourceFile> {
        self.with_db(|db| db.parse(file_id).tree())
    }

    /// Returns true if this file belongs to an immutable library.
    pub fn is_library_file(&self, file_id: FileId) -> Cancellable<bool> {
        use ide_db::base_db::SourceDatabaseExt;
        self.with_db(|db| db.source_root(db.file_source_root(file_id)).is_library)
    }

    /// Gets the file's `LineIndex`: data structure to convert between absolute
    /// offsets and line/column representation.
    pub fn file_line_index(&self, file_id: FileId) -> Cancellable<Arc<LineIndex>> {
        self.with_db(|db| db.line_index(file_id))
    }

    /// Selects the next syntactic nodes encompassing the range.
    pub fn extend_selection(&self, frange: FileRange) -> Cancellable<TextRange> {
        self.with_db(|db| extend_selection::extend_selection(db, frange))
    }

    /// Returns position of the matching brace (all types of braces are
    /// supported).
    pub fn matching_brace(&self, position: FilePosition) -> Cancellable<Option<TextSize>> {
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
    ) -> Cancellable<String> {
        self.with_db(|db| syntax_tree::syntax_tree(db, file_id, text_range))
    }

    pub fn view_hir(&self, position: FilePosition) -> Cancellable<String> {
        self.with_db(|db| view_hir::view_hir(db, position))
    }

    pub fn view_item_tree(&self, file_id: FileId) -> Cancellable<String> {
        self.with_db(|db| view_item_tree::view_item_tree(db, file_id))
    }

    /// Renders the crate graph to GraphViz "dot" syntax.
    pub fn view_crate_graph(&self, full: bool) -> Cancellable<Result<String, String>> {
        self.with_db(|db| view_crate_graph::view_crate_graph(db, full))
    }

    pub fn expand_macro(&self, position: FilePosition) -> Cancellable<Option<ExpandedMacro>> {
        self.with_db(|db| expand_macro::expand_macro(db, position))
    }

    /// Returns an edit to remove all newlines in the range, cleaning up minor
    /// stuff like trailing commas.
    pub fn join_lines(&self, config: &JoinLinesConfig, frange: FileRange) -> Cancellable<TextEdit> {
        self.with_db(|db| {
            let parse = db.parse(frange.file_id);
            join_lines::join_lines(config, &parse.tree(), frange.range)
        })
    }

    /// Returns an edit which should be applied when opening a new line, fixing
    /// up minor stuff like continuing the comment.
    /// The edit will be a snippet (with `$0`).
    pub fn on_enter(&self, position: FilePosition) -> Cancellable<Option<TextEdit>> {
        self.with_db(|db| typing::on_enter(db, position))
    }

    /// Returns an edit which should be applied after a character was typed.
    ///
    /// This is useful for some on-the-fly fixups, like adding `;` to `let =`
    /// automatically.
    pub fn on_char_typed(
        &self,
        position: FilePosition,
        char_typed: char,
        autoclose: bool,
    ) -> Cancellable<Option<SourceChange>> {
        // Fast path to not even parse the file.
        if !typing::TRIGGER_CHARS.contains(char_typed) {
            return Ok(None);
        }
        if char_typed == '<' && !autoclose {
            return Ok(None);
        }

        self.with_db(|db| typing::on_char_typed(db, position, char_typed))
    }

    /// Returns a tree representation of symbols in the file. Useful to draw a
    /// file outline.
    pub fn file_structure(&self, file_id: FileId) -> Cancellable<Vec<StructureNode>> {
        self.with_db(|db| file_structure::file_structure(&db.parse(file_id).tree()))
    }

    /// Returns a list of the places in the file where type hints can be displayed.
    pub fn inlay_hints(
        &self,
        config: &InlayHintsConfig,
        file_id: FileId,
        range: Option<TextRange>,
    ) -> Cancellable<Vec<InlayHint>> {
        self.with_db(|db| inlay_hints::inlay_hints(db, file_id, range, config))
    }

    /// Returns the set of folding ranges.
    pub fn folding_ranges(&self, file_id: FileId) -> Cancellable<Vec<Fold>> {
        self.with_db(|db| folding_ranges::folding_ranges(&db.parse(file_id).tree()))
    }

    /// Fuzzy searches for a symbol.
    pub fn symbol_search(&self, query: Query) -> Cancellable<Vec<NavigationTarget>> {
        self.with_db(|db| {
            symbol_index::world_symbols(db, query)
                .into_iter() // xx: should we make this a par iter?
                .filter_map(|s| s.try_to_nav(db))
                .collect::<Vec<_>>()
        })
    }

    /// Returns the definitions from the symbol at `position`.
    pub fn goto_definition(
        &self,
        position: FilePosition,
    ) -> Cancellable<Option<RangeInfo<Vec<NavigationTarget>>>> {
        self.with_db(|db| goto_definition::goto_definition(db, position))
    }

    /// Returns the declaration from the symbol at `position`.
    pub fn goto_declaration(
        &self,
        position: FilePosition,
    ) -> Cancellable<Option<RangeInfo<Vec<NavigationTarget>>>> {
        self.with_db(|db| goto_declaration::goto_declaration(db, position))
    }

    /// Returns the impls from the symbol at `position`.
    pub fn goto_implementation(
        &self,
        position: FilePosition,
    ) -> Cancellable<Option<RangeInfo<Vec<NavigationTarget>>>> {
        self.with_db(|db| goto_implementation::goto_implementation(db, position))
    }

    /// Returns the type definitions for the symbol at `position`.
    pub fn goto_type_definition(
        &self,
        position: FilePosition,
    ) -> Cancellable<Option<RangeInfo<Vec<NavigationTarget>>>> {
        self.with_db(|db| goto_type_definition::goto_type_definition(db, position))
    }

    /// Finds all usages of the reference at point.
    pub fn find_all_refs(
        &self,
        position: FilePosition,
        search_scope: Option<SearchScope>,
    ) -> Cancellable<Option<Vec<ReferenceSearchResult>>> {
        self.with_db(|db| references::find_all_refs(&Semantics::new(db), position, search_scope))
    }

    /// Returns a short text describing element at position.
    pub fn hover(
        &self,
        config: &HoverConfig,
        range: FileRange,
    ) -> Cancellable<Option<RangeInfo<HoverResult>>> {
        self.with_db(|db| hover::hover(db, range, config))
    }

    /// Returns moniker of symbol at position.
    pub fn moniker(
        &self,
        position: FilePosition,
    ) -> Cancellable<Option<RangeInfo<Vec<moniker::MonikerResult>>>> {
        self.with_db(|db| moniker::moniker(db, position))
    }

    /// Return URL(s) for the documentation of the symbol under the cursor.
    pub fn external_docs(
        &self,
        position: FilePosition,
    ) -> Cancellable<Option<doc_links::DocumentationLink>> {
        self.with_db(|db| doc_links::external_docs(db, &position))
    }

    /// Computes parameter information at the given position.
    pub fn signature_help(&self, position: FilePosition) -> Cancellable<Option<SignatureHelp>> {
        self.with_db(|db| signature_help::signature_help(db, position))
    }

    /// Computes call hierarchy candidates for the given file position.
    pub fn call_hierarchy(
        &self,
        position: FilePosition,
    ) -> Cancellable<Option<RangeInfo<Vec<NavigationTarget>>>> {
        self.with_db(|db| call_hierarchy::call_hierarchy(db, position))
    }

    /// Computes incoming calls for the given file position.
    pub fn incoming_calls(&self, position: FilePosition) -> Cancellable<Option<Vec<CallItem>>> {
        self.with_db(|db| call_hierarchy::incoming_calls(db, position))
    }

    /// Computes outgoing calls for the given file position.
    pub fn outgoing_calls(&self, position: FilePosition) -> Cancellable<Option<Vec<CallItem>>> {
        self.with_db(|db| call_hierarchy::outgoing_calls(db, position))
    }

    /// Returns a `mod name;` declaration which created the current module.
    pub fn parent_module(&self, position: FilePosition) -> Cancellable<Vec<NavigationTarget>> {
        self.with_db(|db| parent_module::parent_module(db, position))
    }

    /// Returns crates this file belongs too.
    pub fn crates_for(&self, file_id: FileId) -> Cancellable<Vec<CrateId>> {
        self.with_db(|db| parent_module::crates_for(db, file_id))
    }

    /// Returns crates this file belongs too.
    pub fn transitive_rev_deps(&self, crate_id: CrateId) -> Cancellable<Vec<CrateId>> {
        self.with_db(|db| db.crate_graph().transitive_rev_deps(crate_id).collect())
    }

    /// Returns crates this file *might* belong too.
    pub fn relevant_crates_for(&self, file_id: FileId) -> Cancellable<Vec<CrateId>> {
        self.with_db(|db| db.relevant_crates(file_id).iter().copied().collect())
    }

    /// Returns the edition of the given crate.
    pub fn crate_edition(&self, crate_id: CrateId) -> Cancellable<Edition> {
        self.with_db(|db| db.crate_graph()[crate_id].edition)
    }

    /// Returns the root file of the given crate.
    pub fn crate_root(&self, crate_id: CrateId) -> Cancellable<FileId> {
        self.with_db(|db| db.crate_graph()[crate_id].root_file_id)
    }

    /// Returns the set of possible targets to run for the current file.
    pub fn runnables(&self, file_id: FileId) -> Cancellable<Vec<Runnable>> {
        self.with_db(|db| runnables::runnables(db, file_id))
    }

    /// Returns the set of tests for the given file position.
    pub fn related_tests(
        &self,
        position: FilePosition,
        search_scope: Option<SearchScope>,
    ) -> Cancellable<Vec<Runnable>> {
        self.with_db(|db| runnables::related_tests(db, position, search_scope))
    }

    /// Computes syntax highlighting for the given file
    pub fn highlight(
        &self,
        highlight_config: HighlightConfig,
        file_id: FileId,
    ) -> Cancellable<Vec<HlRange>> {
        self.with_db(|db| syntax_highlighting::highlight(db, highlight_config, file_id, None))
    }

    /// Computes all ranges to highlight for a given item in a file.
    pub fn highlight_related(
        &self,
        config: HighlightRelatedConfig,
        position: FilePosition,
    ) -> Cancellable<Option<Vec<HighlightedRange>>> {
        self.with_db(|db| {
            highlight_related::highlight_related(&Semantics::new(db), config, position)
        })
    }

    /// Computes syntax highlighting for the given file range.
    pub fn highlight_range(
        &self,
        highlight_config: HighlightConfig,
        frange: FileRange,
    ) -> Cancellable<Vec<HlRange>> {
        self.with_db(|db| {
            syntax_highlighting::highlight(db, highlight_config, frange.file_id, Some(frange.range))
        })
    }

    /// Computes syntax highlighting for the given file.
    pub fn highlight_as_html(&self, file_id: FileId, rainbow: bool) -> Cancellable<String> {
        self.with_db(|db| syntax_highlighting::highlight_as_html(db, file_id, rainbow))
    }

    /// Computes completions at the given position.
    pub fn completions(
        &self,
        config: &CompletionConfig,
        position: FilePosition,
        trigger_character: Option<char>,
    ) -> Cancellable<Option<Vec<CompletionItem>>> {
        self.with_db(|db| {
            ide_completion::completions(db, config, position, trigger_character).map(Into::into)
        })
    }

    /// Resolves additional completion data at the position given.
    pub fn resolve_completion_edits(
        &self,
        config: &CompletionConfig,
        position: FilePosition,
        imports: impl IntoIterator<Item = (String, String)> + std::panic::UnwindSafe,
    ) -> Cancellable<Vec<TextEdit>> {
        Ok(self
            .with_db(|db| ide_completion::resolve_completion_edits(db, config, position, imports))?
            .unwrap_or_default())
    }

    /// Computes the set of diagnostics for the given file.
    pub fn diagnostics(
        &self,
        config: &DiagnosticsConfig,
        resolve: AssistResolveStrategy,
        file_id: FileId,
    ) -> Cancellable<Vec<Diagnostic>> {
        self.with_db(|db| ide_diagnostics::diagnostics(db, config, &resolve, file_id))
    }

    /// Convenience function to return assists + quick fixes for diagnostics
    pub fn assists_with_fixes(
        &self,
        assist_config: &AssistConfig,
        diagnostics_config: &DiagnosticsConfig,
        resolve: AssistResolveStrategy,
        frange: FileRange,
    ) -> Cancellable<Vec<Assist>> {
        let include_fixes = match &assist_config.allowed {
            Some(it) => it.iter().any(|&it| it == AssistKind::None || it == AssistKind::QuickFix),
            None => true,
        };

        self.with_db(|db| {
            let diagnostic_assists = if include_fixes {
                ide_diagnostics::diagnostics(db, diagnostics_config, &resolve, frange.file_id)
                    .into_iter()
                    .flat_map(|it| it.fixes.unwrap_or_default())
                    .filter(|it| it.target.intersect(frange.range).is_some())
                    .collect()
            } else {
                Vec::new()
            };
            let ssr_assists = ssr::ssr_assists(db, &resolve, frange);
            let assists = ide_assists::assists(db, assist_config, resolve, frange);

            let mut res = diagnostic_assists;
            res.extend(ssr_assists.into_iter());
            res.extend(assists.into_iter());

            res
        })
    }

    /// Returns the edit required to rename reference at the position to the new
    /// name.
    pub fn rename(
        &self,
        position: FilePosition,
        new_name: &str,
    ) -> Cancellable<Result<SourceChange, RenameError>> {
        self.with_db(|db| rename::rename(db, position, new_name))
    }

    pub fn prepare_rename(
        &self,
        position: FilePosition,
    ) -> Cancellable<Result<RangeInfo<()>, RenameError>> {
        self.with_db(|db| rename::prepare_rename(db, position))
    }

    pub fn will_rename_file(
        &self,
        file_id: FileId,
        new_name_stem: &str,
    ) -> Cancellable<Option<SourceChange>> {
        self.with_db(|db| rename::will_rename_file(db, file_id, new_name_stem))
    }

    pub fn structural_search_replace(
        &self,
        query: &str,
        parse_only: bool,
        resolve_context: FilePosition,
        selections: Vec<FileRange>,
    ) -> Cancellable<Result<SourceChange, SsrError>> {
        self.with_db(|db| {
            let rule: ide_ssr::SsrRule = query.parse()?;
            let mut match_finder =
                ide_ssr::MatchFinder::in_context(db, resolve_context, selections)?;
            match_finder.add_rule(rule)?;
            let edits = if parse_only { Default::default() } else { match_finder.edits() };
            Ok(SourceChange::from(edits))
        })
    }

    pub fn annotations(
        &self,
        config: &AnnotationConfig,
        file_id: FileId,
    ) -> Cancellable<Vec<Annotation>> {
        self.with_db(|db| annotations::annotations(db, config, file_id))
    }

    pub fn resolve_annotation(&self, annotation: Annotation) -> Cancellable<Annotation> {
        self.with_db(|db| annotations::resolve_annotation(db, annotation))
    }

    pub fn move_item(
        &self,
        range: FileRange,
        direction: Direction,
    ) -> Cancellable<Option<TextEdit>> {
        self.with_db(|db| move_item::move_item(db, range, direction))
    }

    /// Performs an operation on the database that may be canceled.
    ///
    /// rust-analyzer needs to be able to answer semantic questions about the
    /// code while the code is being modified. A common problem is that a
    /// long-running query is being calculated when a new change arrives.
    ///
    /// We can't just apply the change immediately: this will cause the pending
    /// query to see inconsistent state (it will observe an absence of
    /// repeatable read). So what we do is we **cancel** all pending queries
    /// before applying the change.
    ///
    /// Salsa implements cancellation by unwinding with a special value and
    /// catching it on the API boundary.
    fn with_db<F, T>(&self, f: F) -> Cancellable<T>
    where
        F: FnOnce(&RootDatabase) -> T + std::panic::UnwindSafe,
    {
        Cancelled::catch(|| f(&self.db))
    }
}

#[test]
fn analysis_is_send() {
    fn is_send<T: Send>() {}
    is_send::<Analysis>();
}
