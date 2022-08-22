//! This module is responsible for implementing handlers for Language Server
//! Protocol. The majority of requests are fulfilled by calling into the
//! `ide` crate.

use std::{
    io::Write as _,
    process::{self, Stdio},
};

use anyhow::Context;
use ide::{
    AnnotationConfig, AssistKind, AssistResolveStrategy, FileId, FilePosition, FileRange,
    HoverAction, HoverGotoTypeData, Query, RangeInfo, Runnable, RunnableKind, SingleResolve,
    SourceChange, TextEdit,
};
use ide_db::SymbolKind;
use lsp_server::ErrorCode;
use lsp_types::{
    CallHierarchyIncomingCall, CallHierarchyIncomingCallsParams, CallHierarchyItem,
    CallHierarchyOutgoingCall, CallHierarchyOutgoingCallsParams, CallHierarchyPrepareParams,
    CodeLens, CompletionItem, Diagnostic, DiagnosticTag, DocumentFormattingParams, FoldingRange,
    FoldingRangeParams, HoverContents, InlayHint, InlayHintParams, Location, LocationLink,
    NumberOrString, Position, PrepareRenameResponse, Range, RenameParams,
    SemanticTokensDeltaParams, SemanticTokensFullDeltaResult, SemanticTokensParams,
    SemanticTokensRangeParams, SemanticTokensRangeResult, SemanticTokensResult, SymbolInformation,
    SymbolTag, TextDocumentIdentifier, Url, WorkspaceEdit,
};
use project_model::{ManifestPath, ProjectWorkspace, TargetKind};
use serde_json::json;
use stdx::{format_to, never};
use syntax::{algo, ast, AstNode, TextRange, TextSize, T};
use vfs::AbsPathBuf;

use crate::{
    cargo_target_spec::CargoTargetSpec,
    config::{RustfmtConfig, WorkspaceSymbolConfig},
    diff::diff,
    from_proto,
    global_state::{GlobalState, GlobalStateSnapshot},
    line_index::LineEndings,
    lsp_ext::{self, PositionOrRange, ViewCrateGraphParams, WorkspaceSymbolParams},
    lsp_utils::{all_edits_are_disjoint, invalid_params_error},
    to_proto, LspError, Result,
};

pub(crate) fn handle_workspace_reload(state: &mut GlobalState, _: ()) -> Result<()> {
    state.proc_macro_clients.clear();
    state.proc_macro_changed = false;
    state.fetch_workspaces_queue.request_op("reload workspace request".to_string());
    state.fetch_build_data_queue.request_op("reload workspace request".to_string());
    Ok(())
}

pub(crate) fn handle_cancel_flycheck(state: &mut GlobalState, _: ()) -> Result<()> {
    let _p = profile::span("handle_stop_flycheck");
    state.flycheck.iter().for_each(|flycheck| flycheck.cancel());
    Ok(())
}

pub(crate) fn handle_analyzer_status(
    snap: GlobalStateSnapshot,
    params: lsp_ext::AnalyzerStatusParams,
) -> Result<String> {
    let _p = profile::span("handle_analyzer_status");

    let mut buf = String::new();

    let mut file_id = None;
    if let Some(tdi) = params.text_document {
        match from_proto::file_id(&snap, &tdi.uri) {
            Ok(it) => file_id = Some(it),
            Err(_) => format_to!(buf, "file {} not found in vfs", tdi.uri),
        }
    }

    if snap.workspaces.is_empty() {
        buf.push_str("No workspaces\n")
    } else {
        buf.push_str("Workspaces:\n");
        format_to!(
            buf,
            "Loaded {:?} packages across {} workspace{}.\n",
            snap.workspaces.iter().map(|w| w.n_packages()).sum::<usize>(),
            snap.workspaces.len(),
            if snap.workspaces.len() == 1 { "" } else { "s" }
        );
    }
    buf.push_str("\nAnalysis:\n");
    buf.push_str(
        &snap
            .analysis
            .status(file_id)
            .unwrap_or_else(|_| "Analysis retrieval was cancelled".to_owned()),
    );
    Ok(buf)
}

pub(crate) fn handle_memory_usage(state: &mut GlobalState, _: ()) -> Result<String> {
    let _p = profile::span("handle_memory_usage");
    let mut mem = state.analysis_host.per_query_memory_usage();
    mem.push(("Remaining".into(), profile::memory_usage().allocated));

    let mut out = String::new();
    for (name, bytes) in mem {
        format_to!(out, "{:>8} {}\n", bytes, name);
    }
    Ok(out)
}

pub(crate) fn handle_shuffle_crate_graph(state: &mut GlobalState, _: ()) -> Result<()> {
    state.analysis_host.shuffle_crate_graph();
    Ok(())
}

pub(crate) fn handle_syntax_tree(
    snap: GlobalStateSnapshot,
    params: lsp_ext::SyntaxTreeParams,
) -> Result<String> {
    let _p = profile::span("handle_syntax_tree");
    let id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.file_line_index(id)?;
    let text_range = params.range.and_then(|r| from_proto::text_range(&line_index, r).ok());
    let res = snap.analysis.syntax_tree(id, text_range)?;
    Ok(res)
}

pub(crate) fn handle_view_hir(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> Result<String> {
    let _p = profile::span("handle_view_hir");
    let position = from_proto::file_position(&snap, params)?;
    let res = snap.analysis.view_hir(position)?;
    Ok(res)
}

pub(crate) fn handle_view_file_text(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentIdentifier,
) -> Result<String> {
    let file_id = from_proto::file_id(&snap, &params.uri)?;
    Ok(snap.analysis.file_text(file_id)?.to_string())
}

pub(crate) fn handle_view_item_tree(
    snap: GlobalStateSnapshot,
    params: lsp_ext::ViewItemTreeParams,
) -> Result<String> {
    let _p = profile::span("handle_view_item_tree");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let res = snap.analysis.view_item_tree(file_id)?;
    Ok(res)
}

pub(crate) fn handle_view_crate_graph(
    snap: GlobalStateSnapshot,
    params: ViewCrateGraphParams,
) -> Result<String> {
    let _p = profile::span("handle_view_crate_graph");
    let dot = snap.analysis.view_crate_graph(params.full)??;
    Ok(dot)
}

pub(crate) fn handle_expand_macro(
    snap: GlobalStateSnapshot,
    params: lsp_ext::ExpandMacroParams,
) -> Result<Option<lsp_ext::ExpandedMacro>> {
    let _p = profile::span("handle_expand_macro");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;
    let offset = from_proto::offset(&line_index, params.position)?;

    let res = snap.analysis.expand_macro(FilePosition { file_id, offset })?;
    Ok(res.map(|it| lsp_ext::ExpandedMacro { name: it.name, expansion: it.expansion }))
}

pub(crate) fn handle_selection_range(
    snap: GlobalStateSnapshot,
    params: lsp_types::SelectionRangeParams,
) -> Result<Option<Vec<lsp_types::SelectionRange>>> {
    let _p = profile::span("handle_selection_range");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;
    let res: Result<Vec<lsp_types::SelectionRange>> = params
        .positions
        .into_iter()
        .map(|position| {
            let offset = from_proto::offset(&line_index, position)?;
            let mut ranges = Vec::new();
            {
                let mut range = TextRange::new(offset, offset);
                loop {
                    ranges.push(range);
                    let frange = FileRange { file_id, range };
                    let next = snap.analysis.extend_selection(frange)?;
                    if next == range {
                        break;
                    } else {
                        range = next
                    }
                }
            }
            let mut range = lsp_types::SelectionRange {
                range: to_proto::range(&line_index, *ranges.last().unwrap()),
                parent: None,
            };
            for &r in ranges.iter().rev().skip(1) {
                range = lsp_types::SelectionRange {
                    range: to_proto::range(&line_index, r),
                    parent: Some(Box::new(range)),
                }
            }
            Ok(range)
        })
        .collect();

    Ok(Some(res?))
}

pub(crate) fn handle_matching_brace(
    snap: GlobalStateSnapshot,
    params: lsp_ext::MatchingBraceParams,
) -> Result<Vec<Position>> {
    let _p = profile::span("handle_matching_brace");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;
    params
        .positions
        .into_iter()
        .map(|position| {
            let offset = from_proto::offset(&line_index, position);
            offset.map(|offset| {
                let offset = match snap.analysis.matching_brace(FilePosition { file_id, offset }) {
                    Ok(Some(matching_brace_offset)) => matching_brace_offset,
                    Err(_) | Ok(None) => offset,
                };
                to_proto::position(&line_index, offset)
            })
        })
        .collect()
}

pub(crate) fn handle_join_lines(
    snap: GlobalStateSnapshot,
    params: lsp_ext::JoinLinesParams,
) -> Result<Vec<lsp_types::TextEdit>> {
    let _p = profile::span("handle_join_lines");

    let config = snap.config.join_lines();
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;

    let mut res = TextEdit::default();
    for range in params.ranges {
        let range = from_proto::text_range(&line_index, range)?;
        let edit = snap.analysis.join_lines(&config, FileRange { file_id, range })?;
        match res.union(edit) {
            Ok(()) => (),
            Err(_edit) => {
                // just ignore overlapping edits
            }
        }
    }

    Ok(to_proto::text_edit_vec(&line_index, res))
}

pub(crate) fn handle_on_enter(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> Result<Option<Vec<lsp_ext::SnippetTextEdit>>> {
    let _p = profile::span("handle_on_enter");
    let position = from_proto::file_position(&snap, params)?;
    let edit = match snap.analysis.on_enter(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let line_index = snap.file_line_index(position.file_id)?;
    let edit = to_proto::snippet_text_edit_vec(&line_index, true, edit);
    Ok(Some(edit))
}

pub(crate) fn handle_on_type_formatting(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentOnTypeFormattingParams,
) -> Result<Option<Vec<lsp_ext::SnippetTextEdit>>> {
    let _p = profile::span("handle_on_type_formatting");
    let mut position = from_proto::file_position(&snap, params.text_document_position)?;
    let line_index = snap.file_line_index(position.file_id)?;

    // in `ide`, the `on_type` invariant is that
    // `text.char_at(position) == typed_char`.
    position.offset -= TextSize::of('.');
    let char_typed = params.ch.chars().next().unwrap_or('\0');

    let text = snap.analysis.file_text(position.file_id)?;
    if stdx::never!(!text[usize::from(position.offset)..].starts_with(char_typed)) {
        return Ok(None);
    }

    // We have an assist that inserts ` ` after typing `->` in `fn foo() ->{`,
    // but it requires precise cursor positioning to work, and one can't
    // position the cursor with on_type formatting. So, let's just toggle this
    // feature off here, hoping that we'll enable it one day, 😿.
    if char_typed == '>' {
        return Ok(None);
    }

    let edit =
        snap.analysis.on_char_typed(position, char_typed, snap.config.typing_autoclose_angle())?;
    let edit = match edit {
        Some(it) => it,
        None => return Ok(None),
    };

    // This should be a single-file edit
    let (_, text_edit) = edit.source_file_edits.into_iter().next().unwrap();

    let change = to_proto::snippet_text_edit_vec(&line_index, edit.is_snippet, text_edit);
    Ok(Some(change))
}

pub(crate) fn handle_document_symbol(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentSymbolParams,
) -> Result<Option<lsp_types::DocumentSymbolResponse>> {
    let _p = profile::span("handle_document_symbol");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;

    let mut parents: Vec<(lsp_types::DocumentSymbol, Option<usize>)> = Vec::new();

    for symbol in snap.analysis.file_structure(file_id)? {
        let mut tags = Vec::new();
        if symbol.deprecated {
            tags.push(SymbolTag::DEPRECATED)
        };

        #[allow(deprecated)]
        let doc_symbol = lsp_types::DocumentSymbol {
            name: symbol.label,
            detail: symbol.detail,
            kind: to_proto::structure_node_kind(symbol.kind),
            tags: Some(tags),
            deprecated: Some(symbol.deprecated),
            range: to_proto::range(&line_index, symbol.node_range),
            selection_range: to_proto::range(&line_index, symbol.navigation_range),
            children: None,
        };
        parents.push((doc_symbol, symbol.parent));
    }

    // Builds hierarchy from a flat list, in reverse order (so that indices
    // makes sense)
    let document_symbols = {
        let mut acc = Vec::new();
        while let Some((mut node, parent_idx)) = parents.pop() {
            if let Some(children) = &mut node.children {
                children.reverse();
            }
            let parent = match parent_idx {
                None => &mut acc,
                Some(i) => parents[i].0.children.get_or_insert_with(Vec::new),
            };
            parent.push(node);
        }
        acc.reverse();
        acc
    };

    let res = if snap.config.hierarchical_symbols() {
        document_symbols.into()
    } else {
        let url = to_proto::url(&snap, file_id);
        let mut symbol_information = Vec::<SymbolInformation>::new();
        for symbol in document_symbols {
            flatten_document_symbol(&symbol, None, &url, &mut symbol_information);
        }
        symbol_information.into()
    };
    return Ok(Some(res));

    fn flatten_document_symbol(
        symbol: &lsp_types::DocumentSymbol,
        container_name: Option<String>,
        url: &Url,
        res: &mut Vec<SymbolInformation>,
    ) {
        let mut tags = Vec::new();

        #[allow(deprecated)]
        if let Some(true) = symbol.deprecated {
            tags.push(SymbolTag::DEPRECATED)
        }

        #[allow(deprecated)]
        res.push(SymbolInformation {
            name: symbol.name.clone(),
            kind: symbol.kind,
            tags: Some(tags),
            deprecated: symbol.deprecated,
            location: Location::new(url.clone(), symbol.range),
            container_name,
        });

        for child in symbol.children.iter().flatten() {
            flatten_document_symbol(child, Some(symbol.name.clone()), url, res);
        }
    }
}

pub(crate) fn handle_workspace_symbol(
    snap: GlobalStateSnapshot,
    params: WorkspaceSymbolParams,
) -> Result<Option<Vec<SymbolInformation>>> {
    let _p = profile::span("handle_workspace_symbol");

    let config = snap.config.workspace_symbol();
    let (all_symbols, libs) = decide_search_scope_and_kind(&params, &config);
    let limit = config.search_limit;

    let query = {
        let query: String = params.query.chars().filter(|&c| c != '#' && c != '*').collect();
        let mut q = Query::new(query);
        if !all_symbols {
            q.only_types();
        }
        if libs {
            q.libs();
        }
        q.limit(limit);
        q
    };
    let mut res = exec_query(&snap, query)?;
    if res.is_empty() && !all_symbols {
        let mut query = Query::new(params.query);
        query.limit(limit);
        res = exec_query(&snap, query)?;
    }

    return Ok(Some(res));

    fn decide_search_scope_and_kind(
        params: &WorkspaceSymbolParams,
        config: &WorkspaceSymbolConfig,
    ) -> (bool, bool) {
        // Support old-style parsing of markers in the query.
        let mut all_symbols = params.query.contains('#');
        let mut libs = params.query.contains('*');

        // If no explicit marker was set, check request params. If that's also empty
        // use global config.
        if !all_symbols {
            let search_kind = match params.search_kind {
                Some(ref search_kind) => search_kind,
                None => &config.search_kind,
            };
            all_symbols = match search_kind {
                lsp_ext::WorkspaceSymbolSearchKind::OnlyTypes => false,
                lsp_ext::WorkspaceSymbolSearchKind::AllSymbols => true,
            }
        }

        if !libs {
            let search_scope = match params.search_scope {
                Some(ref search_scope) => search_scope,
                None => &config.search_scope,
            };
            libs = match search_scope {
                lsp_ext::WorkspaceSymbolSearchScope::Workspace => false,
                lsp_ext::WorkspaceSymbolSearchScope::WorkspaceAndDependencies => true,
            }
        }

        (all_symbols, libs)
    }

    fn exec_query(snap: &GlobalStateSnapshot, query: Query) -> Result<Vec<SymbolInformation>> {
        let mut res = Vec::new();
        for nav in snap.analysis.symbol_search(query)? {
            let container_name = nav.container_name.as_ref().map(|v| v.to_string());

            #[allow(deprecated)]
            let info = SymbolInformation {
                name: nav.name.to_string(),
                kind: nav
                    .kind
                    .map(to_proto::symbol_kind)
                    .unwrap_or(lsp_types::SymbolKind::VARIABLE),
                tags: None,
                location: to_proto::location_from_nav(snap, nav)?,
                container_name,
                deprecated: None,
            };
            res.push(info);
        }
        Ok(res)
    }
}

pub(crate) fn handle_will_rename_files(
    snap: GlobalStateSnapshot,
    params: lsp_types::RenameFilesParams,
) -> Result<Option<lsp_types::WorkspaceEdit>> {
    let _p = profile::span("handle_will_rename_files");

    let source_changes: Vec<SourceChange> = params
        .files
        .into_iter()
        .filter_map(|file_rename| {
            let from = Url::parse(&file_rename.old_uri).ok()?;
            let to = Url::parse(&file_rename.new_uri).ok()?;

            let from_path = from.to_file_path().ok()?;
            let to_path = to.to_file_path().ok()?;

            // Limit to single-level moves for now.
            match (from_path.parent(), to_path.parent()) {
                (Some(p1), Some(p2)) if p1 == p2 => {
                    if from_path.is_dir() {
                        // add '/' to end of url -- from `file://path/to/folder` to `file://path/to/folder/`
                        let mut old_folder_name = from_path.file_stem()?.to_str()?.to_string();
                        old_folder_name.push('/');
                        let from_with_trailing_slash = from.join(&old_folder_name).ok()?;

                        let imitate_from_url = from_with_trailing_slash.join("mod.rs").ok()?;
                        let new_file_name = to_path.file_name()?.to_str()?;
                        Some((
                            snap.url_to_file_id(&imitate_from_url).ok()?,
                            new_file_name.to_string(),
                        ))
                    } else {
                        let old_name = from_path.file_stem()?.to_str()?;
                        let new_name = to_path.file_stem()?.to_str()?;
                        match (old_name, new_name) {
                            ("mod", _) => None,
                            (_, "mod") => None,
                            _ => Some((snap.url_to_file_id(&from).ok()?, new_name.to_string())),
                        }
                    }
                }
                _ => None,
            }
        })
        .filter_map(|(file_id, new_name)| {
            snap.analysis.will_rename_file(file_id, &new_name).ok()?
        })
        .collect();

    // Drop file system edits since we're just renaming things on the same level
    let mut source_changes = source_changes.into_iter();
    let mut source_change = source_changes.next().unwrap_or_default();
    source_change.file_system_edits.clear();
    // no collect here because we want to merge text edits on same file ids
    source_change.extend(source_changes.flat_map(|it| it.source_file_edits));
    if source_change.source_file_edits.is_empty() {
        Ok(None)
    } else {
        to_proto::workspace_edit(&snap, source_change).map(Some)
    }
}

pub(crate) fn handle_goto_definition(
    snap: GlobalStateSnapshot,
    params: lsp_types::GotoDefinitionParams,
) -> Result<Option<lsp_types::GotoDefinitionResponse>> {
    let _p = profile::span("handle_goto_definition");
    let position = from_proto::file_position(&snap, params.text_document_position_params)?;
    let nav_info = match snap.analysis.goto_definition(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let src = FileRange { file_id: position.file_id, range: nav_info.range };
    let res = to_proto::goto_definition_response(&snap, Some(src), nav_info.info)?;
    Ok(Some(res))
}

pub(crate) fn handle_goto_declaration(
    snap: GlobalStateSnapshot,
    params: lsp_types::request::GotoDeclarationParams,
) -> Result<Option<lsp_types::request::GotoDeclarationResponse>> {
    let _p = profile::span("handle_goto_declaration");
    let position = from_proto::file_position(&snap, params.text_document_position_params.clone())?;
    let nav_info = match snap.analysis.goto_declaration(position)? {
        None => return handle_goto_definition(snap, params),
        Some(it) => it,
    };
    let src = FileRange { file_id: position.file_id, range: nav_info.range };
    let res = to_proto::goto_definition_response(&snap, Some(src), nav_info.info)?;
    Ok(Some(res))
}

pub(crate) fn handle_goto_implementation(
    snap: GlobalStateSnapshot,
    params: lsp_types::request::GotoImplementationParams,
) -> Result<Option<lsp_types::request::GotoImplementationResponse>> {
    let _p = profile::span("handle_goto_implementation");
    let position = from_proto::file_position(&snap, params.text_document_position_params)?;
    let nav_info = match snap.analysis.goto_implementation(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let src = FileRange { file_id: position.file_id, range: nav_info.range };
    let res = to_proto::goto_definition_response(&snap, Some(src), nav_info.info)?;
    Ok(Some(res))
}

pub(crate) fn handle_goto_type_definition(
    snap: GlobalStateSnapshot,
    params: lsp_types::request::GotoTypeDefinitionParams,
) -> Result<Option<lsp_types::request::GotoTypeDefinitionResponse>> {
    let _p = profile::span("handle_goto_type_definition");
    let position = from_proto::file_position(&snap, params.text_document_position_params)?;
    let nav_info = match snap.analysis.goto_type_definition(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let src = FileRange { file_id: position.file_id, range: nav_info.range };
    let res = to_proto::goto_definition_response(&snap, Some(src), nav_info.info)?;
    Ok(Some(res))
}

pub(crate) fn handle_parent_module(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> Result<Option<lsp_types::GotoDefinitionResponse>> {
    let _p = profile::span("handle_parent_module");
    if let Ok(file_path) = &params.text_document.uri.to_file_path() {
        if file_path.file_name().unwrap_or_default() == "Cargo.toml" {
            // search workspaces for parent packages or fallback to workspace root
            let abs_path_buf = match AbsPathBuf::try_from(file_path.to_path_buf()).ok() {
                Some(abs_path_buf) => abs_path_buf,
                None => return Ok(None),
            };

            let manifest_path = match ManifestPath::try_from(abs_path_buf).ok() {
                Some(manifest_path) => manifest_path,
                None => return Ok(None),
            };

            let links: Vec<LocationLink> = snap
                .workspaces
                .iter()
                .filter_map(|ws| match ws {
                    ProjectWorkspace::Cargo { cargo, .. } => cargo.parent_manifests(&manifest_path),
                    _ => None,
                })
                .flatten()
                .map(|parent_manifest_path| LocationLink {
                    origin_selection_range: None,
                    target_uri: to_proto::url_from_abs_path(&parent_manifest_path),
                    target_range: Range::default(),
                    target_selection_range: Range::default(),
                })
                .collect::<_>();
            return Ok(Some(links.into()));
        }

        // check if invoked at the crate root
        let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
        let crate_id = match snap.analysis.crate_for(file_id)?.first() {
            Some(&crate_id) => crate_id,
            None => return Ok(None),
        };
        let cargo_spec = match CargoTargetSpec::for_file(&snap, file_id)? {
            Some(it) => it,
            None => return Ok(None),
        };

        if snap.analysis.crate_root(crate_id)? == file_id {
            let cargo_toml_url = to_proto::url_from_abs_path(&cargo_spec.cargo_toml);
            let res = vec![LocationLink {
                origin_selection_range: None,
                target_uri: cargo_toml_url,
                target_range: Range::default(),
                target_selection_range: Range::default(),
            }]
            .into();
            return Ok(Some(res));
        }
    }

    // locate parent module by semantics
    let position = from_proto::file_position(&snap, params)?;
    let navs = snap.analysis.parent_module(position)?;
    let res = to_proto::goto_definition_response(&snap, None, navs)?;
    Ok(Some(res))
}

pub(crate) fn handle_runnables(
    snap: GlobalStateSnapshot,
    params: lsp_ext::RunnablesParams,
) -> Result<Vec<lsp_ext::Runnable>> {
    let _p = profile::span("handle_runnables");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;
    let offset = params.position.and_then(|it| from_proto::offset(&line_index, it).ok());
    let cargo_spec = CargoTargetSpec::for_file(&snap, file_id)?;

    let expect_test = match offset {
        Some(offset) => {
            let source_file = snap.analysis.parse(file_id)?;
            algo::find_node_at_offset::<ast::MacroCall>(source_file.syntax(), offset)
                .and_then(|it| it.path()?.segment()?.name_ref())
                .map_or(false, |it| it.text() == "expect" || it.text() == "expect_file")
        }
        None => false,
    };

    let mut res = Vec::new();
    for runnable in snap.analysis.runnables(file_id)? {
        if should_skip_for_offset(&runnable, offset) {
            continue;
        }
        if should_skip_target(&runnable, cargo_spec.as_ref()) {
            continue;
        }
        let mut runnable = to_proto::runnable(&snap, runnable)?;
        if expect_test {
            runnable.label = format!("{} + expect", runnable.label);
            runnable.args.expect_test = Some(true);
        }
        res.push(runnable);
    }

    // Add `cargo check` and `cargo test` for all targets of the whole package
    let config = snap.config.runnables();
    match cargo_spec {
        Some(spec) => {
            for cmd in ["check", "test"] {
                res.push(lsp_ext::Runnable {
                    label: format!("cargo {} -p {} --all-targets", cmd, spec.package),
                    location: None,
                    kind: lsp_ext::RunnableKind::Cargo,
                    args: lsp_ext::CargoRunnable {
                        workspace_root: Some(spec.workspace_root.clone().into()),
                        override_cargo: config.override_cargo.clone(),
                        cargo_args: vec![
                            cmd.to_string(),
                            "--package".to_string(),
                            spec.package.clone(),
                            "--all-targets".to_string(),
                        ],
                        cargo_extra_args: config.cargo_extra_args.clone(),
                        executable_args: Vec::new(),
                        expect_test: None,
                    },
                })
            }
        }
        None => {
            if !snap.config.linked_projects().is_empty()
                || !snap
                    .config
                    .discovered_projects
                    .as_ref()
                    .map(|projects| projects.is_empty())
                    .unwrap_or(true)
            {
                res.push(lsp_ext::Runnable {
                    label: "cargo check --workspace".to_string(),
                    location: None,
                    kind: lsp_ext::RunnableKind::Cargo,
                    args: lsp_ext::CargoRunnable {
                        workspace_root: None,
                        override_cargo: config.override_cargo,
                        cargo_args: vec!["check".to_string(), "--workspace".to_string()],
                        cargo_extra_args: config.cargo_extra_args,
                        executable_args: Vec::new(),
                        expect_test: None,
                    },
                });
            }
        }
    }
    Ok(res)
}

fn should_skip_for_offset(runnable: &Runnable, offset: Option<TextSize>) -> bool {
    match offset {
        None => false,
        _ if matches!(&runnable.kind, RunnableKind::TestMod { .. }) => false,
        Some(offset) => !runnable.nav.full_range.contains_inclusive(offset),
    }
}

pub(crate) fn handle_related_tests(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> Result<Vec<lsp_ext::TestInfo>> {
    let _p = profile::span("handle_related_tests");
    let position = from_proto::file_position(&snap, params)?;

    let tests = snap.analysis.related_tests(position, None)?;
    let mut res = Vec::new();
    for it in tests {
        if let Ok(runnable) = to_proto::runnable(&snap, it) {
            res.push(lsp_ext::TestInfo { runnable })
        }
    }

    Ok(res)
}

pub(crate) fn handle_completion(
    snap: GlobalStateSnapshot,
    params: lsp_types::CompletionParams,
) -> Result<Option<lsp_types::CompletionResponse>> {
    let _p = profile::span("handle_completion");
    let text_document_position = params.text_document_position.clone();
    let position = from_proto::file_position(&snap, params.text_document_position)?;
    let completion_trigger_character =
        params.context.and_then(|ctx| ctx.trigger_character).and_then(|s| s.chars().next());

    if Some(':') == completion_trigger_character {
        let source_file = snap.analysis.parse(position.file_id)?;
        let left_token = source_file.syntax().token_at_offset(position.offset).left_biased();
        let completion_triggered_after_single_colon = match left_token {
            Some(left_token) => left_token.kind() == T![:],
            None => true,
        };
        if completion_triggered_after_single_colon {
            return Ok(None);
        }
    }

    let completion_config = &snap.config.completion();
    let items = match snap.analysis.completions(
        completion_config,
        position,
        completion_trigger_character,
    )? {
        None => return Ok(None),
        Some(items) => items,
    };
    let line_index = snap.file_line_index(position.file_id)?;

    let items =
        to_proto::completion_items(&snap.config, &line_index, text_document_position, items);

    let completion_list = lsp_types::CompletionList { is_incomplete: true, items };
    Ok(Some(completion_list.into()))
}

pub(crate) fn handle_completion_resolve(
    snap: GlobalStateSnapshot,
    mut original_completion: CompletionItem,
) -> Result<CompletionItem> {
    let _p = profile::span("handle_completion_resolve");

    if !all_edits_are_disjoint(&original_completion, &[]) {
        return Err(invalid_params_error(
            "Received a completion with overlapping edits, this is not LSP-compliant".to_string(),
        )
        .into());
    }

    let data = match original_completion.data.take() {
        Some(it) => it,
        None => return Ok(original_completion),
    };

    let resolve_data: lsp_ext::CompletionResolveData = serde_json::from_value(data)?;

    let file_id = from_proto::file_id(&snap, &resolve_data.position.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;
    let offset = from_proto::offset(&line_index, resolve_data.position.position)?;

    let additional_edits = snap
        .analysis
        .resolve_completion_edits(
            &snap.config.completion(),
            FilePosition { file_id, offset },
            resolve_data
                .imports
                .into_iter()
                .map(|import| (import.full_import_path, import.imported_name)),
        )?
        .into_iter()
        .flat_map(|edit| edit.into_iter().map(|indel| to_proto::text_edit(&line_index, indel)))
        .collect::<Vec<_>>();

    if !all_edits_are_disjoint(&original_completion, &additional_edits) {
        return Err(LspError::new(
            ErrorCode::InternalError as i32,
            "Import edit overlaps with the original completion edits, this is not LSP-compliant"
                .into(),
        )
        .into());
    }

    if let Some(original_additional_edits) = original_completion.additional_text_edits.as_mut() {
        original_additional_edits.extend(additional_edits.into_iter())
    } else {
        original_completion.additional_text_edits = Some(additional_edits);
    }

    Ok(original_completion)
}

pub(crate) fn handle_folding_range(
    snap: GlobalStateSnapshot,
    params: FoldingRangeParams,
) -> Result<Option<Vec<FoldingRange>>> {
    let _p = profile::span("handle_folding_range");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let folds = snap.analysis.folding_ranges(file_id)?;
    let text = snap.analysis.file_text(file_id)?;
    let line_index = snap.file_line_index(file_id)?;
    let line_folding_only = snap.config.line_folding_only();
    let res = folds
        .into_iter()
        .map(|it| to_proto::folding_range(&*text, &line_index, line_folding_only, it))
        .collect();
    Ok(Some(res))
}

pub(crate) fn handle_signature_help(
    snap: GlobalStateSnapshot,
    params: lsp_types::SignatureHelpParams,
) -> Result<Option<lsp_types::SignatureHelp>> {
    let _p = profile::span("handle_signature_help");
    let position = from_proto::file_position(&snap, params.text_document_position_params)?;
    let help = match snap.analysis.signature_help(position)? {
        Some(it) => it,
        None => return Ok(None),
    };
    let config = snap.config.call_info();
    let res = to_proto::signature_help(help, config, snap.config.signature_help_label_offsets());
    Ok(Some(res))
}

pub(crate) fn handle_hover(
    snap: GlobalStateSnapshot,
    params: lsp_ext::HoverParams,
) -> Result<Option<lsp_ext::Hover>> {
    let _p = profile::span("handle_hover");
    let range = match params.position {
        PositionOrRange::Position(position) => Range::new(position, position),
        PositionOrRange::Range(range) => range,
    };

    let file_range = from_proto::file_range(&snap, params.text_document, range)?;
    let info = match snap.analysis.hover(&snap.config.hover(), file_range)? {
        None => return Ok(None),
        Some(info) => info,
    };

    let line_index = snap.file_line_index(file_range.file_id)?;
    let range = to_proto::range(&line_index, info.range);
    let markup_kind =
        snap.config.hover().documentation.map_or(ide::HoverDocFormat::Markdown, |kind| kind);
    let hover = lsp_ext::Hover {
        hover: lsp_types::Hover {
            contents: HoverContents::Markup(to_proto::markup_content(
                info.info.markup,
                markup_kind,
            )),
            range: Some(range),
        },
        actions: if snap.config.hover_actions().none() {
            Vec::new()
        } else {
            prepare_hover_actions(&snap, &info.info.actions)
        },
    };

    Ok(Some(hover))
}

pub(crate) fn handle_prepare_rename(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> Result<Option<PrepareRenameResponse>> {
    let _p = profile::span("handle_prepare_rename");
    let position = from_proto::file_position(&snap, params)?;

    let change = snap.analysis.prepare_rename(position)?.map_err(to_proto::rename_error)?;

    let line_index = snap.file_line_index(position.file_id)?;
    let range = to_proto::range(&line_index, change.range);
    Ok(Some(PrepareRenameResponse::Range(range)))
}

pub(crate) fn handle_rename(
    snap: GlobalStateSnapshot,
    params: RenameParams,
) -> Result<Option<WorkspaceEdit>> {
    let _p = profile::span("handle_rename");
    let position = from_proto::file_position(&snap, params.text_document_position)?;

    let mut change =
        snap.analysis.rename(position, &*params.new_name)?.map_err(to_proto::rename_error)?;

    // this is kind of a hack to prevent double edits from happening when moving files
    // When a module gets renamed by renaming the mod declaration this causes the file to move
    // which in turn will trigger a WillRenameFiles request to the server for which we reply with a
    // a second identical set of renames, the client will then apply both edits causing incorrect edits
    // with this we only emit source_file_edits in the WillRenameFiles response which will do the rename instead
    // See https://github.com/microsoft/vscode-languageserver-node/issues/752 for more info
    if !change.file_system_edits.is_empty() && snap.config.will_rename() {
        change.source_file_edits.clear();
    }
    let workspace_edit = to_proto::workspace_edit(&snap, change)?;
    Ok(Some(workspace_edit))
}

pub(crate) fn handle_references(
    snap: GlobalStateSnapshot,
    params: lsp_types::ReferenceParams,
) -> Result<Option<Vec<Location>>> {
    let _p = profile::span("handle_references");
    let position = from_proto::file_position(&snap, params.text_document_position)?;

    let refs = match snap.analysis.find_all_refs(position, None)? {
        None => return Ok(None),
        Some(refs) => refs,
    };

    let include_declaration = params.context.include_declaration;
    let locations = refs
        .into_iter()
        .flat_map(|refs| {
            let decl = if include_declaration {
                refs.declaration.map(|decl| FileRange {
                    file_id: decl.nav.file_id,
                    range: decl.nav.focus_or_full_range(),
                })
            } else {
                None
            };
            refs.references
                .into_iter()
                .flat_map(|(file_id, refs)| {
                    refs.into_iter().map(move |(range, _)| FileRange { file_id, range })
                })
                .chain(decl)
        })
        .filter_map(|frange| to_proto::location(&snap, frange).ok())
        .collect();

    Ok(Some(locations))
}

pub(crate) fn handle_formatting(
    snap: GlobalStateSnapshot,
    params: DocumentFormattingParams,
) -> Result<Option<Vec<lsp_types::TextEdit>>> {
    let _p = profile::span("handle_formatting");

    run_rustfmt(&snap, params.text_document, None)
}

pub(crate) fn handle_range_formatting(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentRangeFormattingParams,
) -> Result<Option<Vec<lsp_types::TextEdit>>> {
    let _p = profile::span("handle_range_formatting");

    run_rustfmt(&snap, params.text_document, Some(params.range))
}

pub(crate) fn handle_code_action(
    snap: GlobalStateSnapshot,
    params: lsp_types::CodeActionParams,
) -> Result<Option<Vec<lsp_ext::CodeAction>>> {
    let _p = profile::span("handle_code_action");

    if !snap.config.code_action_literals() {
        // We intentionally don't support command-based actions, as those either
        // require either custom client-code or server-initiated edits. Server
        // initiated edits break causality, so we avoid those.
        return Ok(None);
    }

    let line_index =
        snap.file_line_index(from_proto::file_id(&snap, &params.text_document.uri)?)?;
    let frange = from_proto::file_range(&snap, params.text_document.clone(), params.range)?;

    let mut assists_config = snap.config.assist();
    assists_config.allowed = params
        .context
        .only
        .clone()
        .map(|it| it.into_iter().filter_map(from_proto::assist_kind).collect());

    let mut res: Vec<lsp_ext::CodeAction> = Vec::new();

    let code_action_resolve_cap = snap.config.code_action_resolve();
    let resolve = if code_action_resolve_cap {
        AssistResolveStrategy::None
    } else {
        AssistResolveStrategy::All
    };
    let assists = snap.analysis.assists_with_fixes(
        &assists_config,
        &snap.config.diagnostics(),
        resolve,
        frange,
    )?;
    for (index, assist) in assists.into_iter().enumerate() {
        let resolve_data =
            if code_action_resolve_cap { Some((index, params.clone())) } else { None };
        let code_action = to_proto::code_action(&snap, assist, resolve_data)?;
        res.push(code_action)
    }

    // Fixes from `cargo check`.
    for fix in
        snap.check_fixes.values().filter_map(|it| it.get(&frange.file_id)).into_iter().flatten()
    {
        // FIXME: this mapping is awkward and shouldn't exist. Refactor
        // `snap.check_fixes` to not convert to LSP prematurely.
        let intersect_fix_range = fix
            .ranges
            .iter()
            .copied()
            .filter_map(|range| from_proto::text_range(&line_index, range).ok())
            .any(|fix_range| fix_range.intersect(frange.range).is_some());
        if intersect_fix_range {
            res.push(fix.action.clone());
        }
    }

    Ok(Some(res))
}

pub(crate) fn handle_code_action_resolve(
    snap: GlobalStateSnapshot,
    mut code_action: lsp_ext::CodeAction,
) -> Result<lsp_ext::CodeAction> {
    let _p = profile::span("handle_code_action_resolve");
    let params = match code_action.data.take() {
        Some(it) => it,
        None => return Err(invalid_params_error("code action without data".to_string()).into()),
    };

    let file_id = from_proto::file_id(&snap, &params.code_action_params.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;
    let range = from_proto::text_range(&line_index, params.code_action_params.range)?;
    let frange = FileRange { file_id, range };

    let mut assists_config = snap.config.assist();
    assists_config.allowed = params
        .code_action_params
        .context
        .only
        .map(|it| it.into_iter().filter_map(from_proto::assist_kind).collect());

    let (assist_index, assist_resolve) = match parse_action_id(&params.id) {
        Ok(parsed_data) => parsed_data,
        Err(e) => {
            return Err(invalid_params_error(format!(
                "Failed to parse action id string '{}': {}",
                params.id, e
            ))
            .into())
        }
    };

    let expected_assist_id = assist_resolve.assist_id.clone();
    let expected_kind = assist_resolve.assist_kind;

    let assists = snap.analysis.assists_with_fixes(
        &assists_config,
        &snap.config.diagnostics(),
        AssistResolveStrategy::Single(assist_resolve),
        frange,
    )?;

    let assist = match assists.get(assist_index) {
        Some(assist) => assist,
        None => return Err(invalid_params_error(format!(
            "Failed to find the assist for index {} provided by the resolve request. Resolve request assist id: {}",
            assist_index, params.id,
        ))
        .into())
    };
    if assist.id.0 != expected_assist_id || assist.id.1 != expected_kind {
        return Err(invalid_params_error(format!(
            "Mismatching assist at index {} for the resolve parameters given. Resolve request assist id: {}, actual id: {:?}.",
            assist_index, params.id, assist.id
        ))
        .into());
    }
    let ca = to_proto::code_action(&snap, assist.clone(), None)?;
    code_action.edit = ca.edit;
    code_action.command = ca.command;
    Ok(code_action)
}

fn parse_action_id(action_id: &str) -> Result<(usize, SingleResolve), String> {
    let id_parts = action_id.split(':').collect::<Vec<_>>();
    match id_parts.as_slice() {
        [assist_id_string, assist_kind_string, index_string] => {
            let assist_kind: AssistKind = assist_kind_string.parse()?;
            let index: usize = match index_string.parse() {
                Ok(index) => index,
                Err(e) => return Err(format!("Incorrect index string: {}", e)),
            };
            Ok((index, SingleResolve { assist_id: assist_id_string.to_string(), assist_kind }))
        }
        _ => Err("Action id contains incorrect number of segments".to_string()),
    }
}

pub(crate) fn handle_code_lens(
    snap: GlobalStateSnapshot,
    params: lsp_types::CodeLensParams,
) -> Result<Option<Vec<CodeLens>>> {
    let _p = profile::span("handle_code_lens");

    let lens_config = snap.config.lens();
    if lens_config.none() {
        // early return before any db query!
        return Ok(Some(Vec::default()));
    }

    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let cargo_target_spec = CargoTargetSpec::for_file(&snap, file_id)?;

    let annotations = snap.analysis.annotations(
        &AnnotationConfig {
            binary_target: cargo_target_spec
                .map(|spec| {
                    matches!(
                        spec.target_kind,
                        TargetKind::Bin | TargetKind::Example | TargetKind::Test
                    )
                })
                .unwrap_or(false),
            annotate_runnables: lens_config.runnable(),
            annotate_impls: lens_config.implementations,
            annotate_references: lens_config.refs_adt,
            annotate_method_references: lens_config.method_refs,
            annotate_enum_variant_references: lens_config.enum_variant_refs,
        },
        file_id,
    )?;

    let mut res = Vec::new();
    for a in annotations {
        to_proto::code_lens(&mut res, &snap, a)?;
    }

    Ok(Some(res))
}

pub(crate) fn handle_code_lens_resolve(
    snap: GlobalStateSnapshot,
    code_lens: CodeLens,
) -> Result<CodeLens> {
    let annotation = from_proto::annotation(&snap, code_lens.clone())?;
    let annotation = snap.analysis.resolve_annotation(annotation)?;

    let mut acc = Vec::new();
    to_proto::code_lens(&mut acc, &snap, annotation)?;

    let res = match acc.pop() {
        Some(it) if acc.is_empty() => it,
        _ => {
            never!();
            code_lens
        }
    };

    Ok(res)
}

pub(crate) fn handle_document_highlight(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentHighlightParams,
) -> Result<Option<Vec<lsp_types::DocumentHighlight>>> {
    let _p = profile::span("handle_document_highlight");
    let position = from_proto::file_position(&snap, params.text_document_position_params)?;
    let line_index = snap.file_line_index(position.file_id)?;

    let refs = match snap.analysis.highlight_related(snap.config.highlight_related(), position)? {
        None => return Ok(None),
        Some(refs) => refs,
    };
    let res = refs
        .into_iter()
        .map(|ide::HighlightedRange { range, category }| lsp_types::DocumentHighlight {
            range: to_proto::range(&line_index, range),
            kind: category.map(to_proto::document_highlight_kind),
        })
        .collect();
    Ok(Some(res))
}

pub(crate) fn handle_ssr(
    snap: GlobalStateSnapshot,
    params: lsp_ext::SsrParams,
) -> Result<lsp_types::WorkspaceEdit> {
    let _p = profile::span("handle_ssr");
    let selections = params
        .selections
        .iter()
        .map(|range| from_proto::file_range(&snap, params.position.text_document.clone(), *range))
        .collect::<Result<Vec<_>, _>>()?;
    let position = from_proto::file_position(&snap, params.position)?;
    let source_change = snap.analysis.structural_search_replace(
        &params.query,
        params.parse_only,
        position,
        selections,
    )??;
    to_proto::workspace_edit(&snap, source_change)
}

pub(crate) fn publish_diagnostics(
    snap: &GlobalStateSnapshot,
    file_id: FileId,
) -> Result<Vec<Diagnostic>> {
    let _p = profile::span("publish_diagnostics");
    let line_index = snap.file_line_index(file_id)?;

    let diagnostics: Vec<Diagnostic> = snap
        .analysis
        .diagnostics(&snap.config.diagnostics(), AssistResolveStrategy::None, file_id)?
        .into_iter()
        .map(|d| Diagnostic {
            range: to_proto::range(&line_index, d.range),
            severity: Some(to_proto::diagnostic_severity(d.severity)),
            code: Some(NumberOrString::String(d.code.as_str().to_string())),
            code_description: Some(lsp_types::CodeDescription {
                href: lsp_types::Url::parse(&format!(
                    "https://rust-analyzer.github.io/manual.html#{}",
                    d.code.as_str()
                ))
                .unwrap(),
            }),
            source: Some("rust-analyzer".to_string()),
            message: d.message,
            related_information: None,
            tags: if d.unused { Some(vec![DiagnosticTag::UNNECESSARY]) } else { None },
            data: None,
        })
        .collect();
    Ok(diagnostics)
}

pub(crate) fn handle_inlay_hints(
    snap: GlobalStateSnapshot,
    params: InlayHintParams,
) -> Result<Option<Vec<InlayHint>>> {
    let _p = profile::span("handle_inlay_hints");
    let document_uri = &params.text_document.uri;
    let file_id = from_proto::file_id(&snap, document_uri)?;
    let line_index = snap.file_line_index(file_id)?;
    let range = from_proto::file_range(
        &snap,
        TextDocumentIdentifier::new(document_uri.to_owned()),
        params.range,
    )?;
    let inlay_hints_config = snap.config.inlay_hints();
    Ok(Some(
        snap.analysis
            .inlay_hints(&inlay_hints_config, file_id, Some(range))?
            .into_iter()
            .map(|it| {
                to_proto::inlay_hint(&snap, &line_index, inlay_hints_config.render_colons, it)
            })
            .collect(),
    ))
}

pub(crate) fn handle_inlay_hints_resolve(
    snap: GlobalStateSnapshot,
    mut hint: InlayHint,
) -> Result<InlayHint> {
    let _p = profile::span("handle_inlay_hints_resolve");
    let data = match hint.data.take() {
        Some(it) => it,
        None => return Ok(hint),
    };

    let resolve_data: lsp_ext::InlayHintResolveData = serde_json::from_value(data)?;

    let file_range = from_proto::file_range(
        &snap,
        resolve_data.text_document,
        match resolve_data.position {
            PositionOrRange::Position(pos) => Range::new(pos, pos),
            PositionOrRange::Range(range) => range,
        },
    )?;
    let info = match snap.analysis.hover(&snap.config.hover(), file_range)? {
        None => return Ok(hint),
        Some(info) => info,
    };

    let markup_kind =
        snap.config.hover().documentation.map_or(ide::HoverDocFormat::Markdown, |kind| kind);

    // FIXME: hover actions?
    hint.tooltip = Some(lsp_types::InlayHintTooltip::MarkupContent(to_proto::markup_content(
        info.info.markup,
        markup_kind,
    )));
    Ok(hint)
}

pub(crate) fn handle_call_hierarchy_prepare(
    snap: GlobalStateSnapshot,
    params: CallHierarchyPrepareParams,
) -> Result<Option<Vec<CallHierarchyItem>>> {
    let _p = profile::span("handle_call_hierarchy_prepare");
    let position = from_proto::file_position(&snap, params.text_document_position_params)?;

    let nav_info = match snap.analysis.call_hierarchy(position)? {
        None => return Ok(None),
        Some(it) => it,
    };

    let RangeInfo { range: _, info: navs } = nav_info;
    let res = navs
        .into_iter()
        .filter(|it| it.kind == Some(SymbolKind::Function))
        .map(|it| to_proto::call_hierarchy_item(&snap, it))
        .collect::<Result<Vec<_>>>()?;

    Ok(Some(res))
}

pub(crate) fn handle_call_hierarchy_incoming(
    snap: GlobalStateSnapshot,
    params: CallHierarchyIncomingCallsParams,
) -> Result<Option<Vec<CallHierarchyIncomingCall>>> {
    let _p = profile::span("handle_call_hierarchy_incoming");
    let item = params.item;

    let doc = TextDocumentIdentifier::new(item.uri);
    let frange = from_proto::file_range(&snap, doc, item.selection_range)?;
    let fpos = FilePosition { file_id: frange.file_id, offset: frange.range.start() };

    let call_items = match snap.analysis.incoming_calls(fpos)? {
        None => return Ok(None),
        Some(it) => it,
    };

    let mut res = vec![];

    for call_item in call_items.into_iter() {
        let file_id = call_item.target.file_id;
        let line_index = snap.file_line_index(file_id)?;
        let item = to_proto::call_hierarchy_item(&snap, call_item.target)?;
        res.push(CallHierarchyIncomingCall {
            from: item,
            from_ranges: call_item
                .ranges
                .into_iter()
                .map(|it| to_proto::range(&line_index, it))
                .collect(),
        });
    }

    Ok(Some(res))
}

pub(crate) fn handle_call_hierarchy_outgoing(
    snap: GlobalStateSnapshot,
    params: CallHierarchyOutgoingCallsParams,
) -> Result<Option<Vec<CallHierarchyOutgoingCall>>> {
    let _p = profile::span("handle_call_hierarchy_outgoing");
    let item = params.item;

    let doc = TextDocumentIdentifier::new(item.uri);
    let frange = from_proto::file_range(&snap, doc, item.selection_range)?;
    let fpos = FilePosition { file_id: frange.file_id, offset: frange.range.start() };

    let call_items = match snap.analysis.outgoing_calls(fpos)? {
        None => return Ok(None),
        Some(it) => it,
    };

    let mut res = vec![];

    for call_item in call_items.into_iter() {
        let file_id = call_item.target.file_id;
        let line_index = snap.file_line_index(file_id)?;
        let item = to_proto::call_hierarchy_item(&snap, call_item.target)?;
        res.push(CallHierarchyOutgoingCall {
            to: item,
            from_ranges: call_item
                .ranges
                .into_iter()
                .map(|it| to_proto::range(&line_index, it))
                .collect(),
        });
    }

    Ok(Some(res))
}

pub(crate) fn handle_semantic_tokens_full(
    snap: GlobalStateSnapshot,
    params: SemanticTokensParams,
) -> Result<Option<SemanticTokensResult>> {
    let _p = profile::span("handle_semantic_tokens_full");

    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let text = snap.analysis.file_text(file_id)?;
    let line_index = snap.file_line_index(file_id)?;

    let highlights = snap.analysis.highlight(snap.config.highlighting_config(), file_id)?;
    let semantic_tokens = to_proto::semantic_tokens(&text, &line_index, highlights);

    // Unconditionally cache the tokens
    snap.semantic_tokens_cache.lock().insert(params.text_document.uri, semantic_tokens.clone());

    Ok(Some(semantic_tokens.into()))
}

pub(crate) fn handle_semantic_tokens_full_delta(
    snap: GlobalStateSnapshot,
    params: SemanticTokensDeltaParams,
) -> Result<Option<SemanticTokensFullDeltaResult>> {
    let _p = profile::span("handle_semantic_tokens_full_delta");

    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let text = snap.analysis.file_text(file_id)?;
    let line_index = snap.file_line_index(file_id)?;

    let highlights = snap.analysis.highlight(snap.config.highlighting_config(), file_id)?;
    let semantic_tokens = to_proto::semantic_tokens(&text, &line_index, highlights);

    let mut cache = snap.semantic_tokens_cache.lock();
    let cached_tokens = cache.entry(params.text_document.uri).or_default();

    if let Some(prev_id) = &cached_tokens.result_id {
        if *prev_id == params.previous_result_id {
            let delta = to_proto::semantic_token_delta(cached_tokens, &semantic_tokens);
            *cached_tokens = semantic_tokens;
            return Ok(Some(delta.into()));
        }
    }

    *cached_tokens = semantic_tokens.clone();

    Ok(Some(semantic_tokens.into()))
}

pub(crate) fn handle_semantic_tokens_range(
    snap: GlobalStateSnapshot,
    params: SemanticTokensRangeParams,
) -> Result<Option<SemanticTokensRangeResult>> {
    let _p = profile::span("handle_semantic_tokens_range");

    let frange = from_proto::file_range(&snap, params.text_document, params.range)?;
    let text = snap.analysis.file_text(frange.file_id)?;
    let line_index = snap.file_line_index(frange.file_id)?;

    let highlights = snap.analysis.highlight_range(snap.config.highlighting_config(), frange)?;
    let semantic_tokens = to_proto::semantic_tokens(&text, &line_index, highlights);
    Ok(Some(semantic_tokens.into()))
}

pub(crate) fn handle_open_docs(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> Result<Option<lsp_types::Url>> {
    let _p = profile::span("handle_open_docs");
    let position = from_proto::file_position(&snap, params)?;

    let remote = snap.analysis.external_docs(position)?;

    Ok(remote.and_then(|remote| Url::parse(&remote).ok()))
}

pub(crate) fn handle_open_cargo_toml(
    snap: GlobalStateSnapshot,
    params: lsp_ext::OpenCargoTomlParams,
) -> Result<Option<lsp_types::GotoDefinitionResponse>> {
    let _p = profile::span("handle_open_cargo_toml");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;

    let cargo_spec = match CargoTargetSpec::for_file(&snap, file_id)? {
        Some(it) => it,
        None => return Ok(None),
    };

    let cargo_toml_url = to_proto::url_from_abs_path(&cargo_spec.cargo_toml);
    let res: lsp_types::GotoDefinitionResponse =
        Location::new(cargo_toml_url, Range::default()).into();
    Ok(Some(res))
}

pub(crate) fn handle_move_item(
    snap: GlobalStateSnapshot,
    params: lsp_ext::MoveItemParams,
) -> Result<Vec<lsp_ext::SnippetTextEdit>> {
    let _p = profile::span("handle_move_item");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let range = from_proto::file_range(&snap, params.text_document, params.range)?;

    let direction = match params.direction {
        lsp_ext::MoveItemDirection::Up => ide::Direction::Up,
        lsp_ext::MoveItemDirection::Down => ide::Direction::Down,
    };

    match snap.analysis.move_item(range, direction)? {
        Some(text_edit) => {
            let line_index = snap.file_line_index(file_id)?;
            Ok(to_proto::snippet_text_edit_vec(&line_index, true, text_edit))
        }
        None => Ok(vec![]),
    }
}

fn to_command_link(command: lsp_types::Command, tooltip: String) -> lsp_ext::CommandLink {
    lsp_ext::CommandLink { tooltip: Some(tooltip), command }
}

fn show_impl_command_link(
    snap: &GlobalStateSnapshot,
    position: &FilePosition,
) -> Option<lsp_ext::CommandLinkGroup> {
    if snap.config.hover_actions().implementations && snap.config.client_commands().show_reference {
        if let Some(nav_data) = snap.analysis.goto_implementation(*position).unwrap_or(None) {
            let uri = to_proto::url(snap, position.file_id);
            let line_index = snap.file_line_index(position.file_id).ok()?;
            let position = to_proto::position(&line_index, position.offset);
            let locations: Vec<_> = nav_data
                .info
                .into_iter()
                .filter_map(|nav| to_proto::location_from_nav(snap, nav).ok())
                .collect();
            let title = to_proto::implementation_title(locations.len());
            let command = to_proto::command::show_references(title, &uri, position, locations);

            return Some(lsp_ext::CommandLinkGroup {
                commands: vec![to_command_link(command, "Go to implementations".into())],
                ..Default::default()
            });
        }
    }
    None
}

fn show_ref_command_link(
    snap: &GlobalStateSnapshot,
    position: &FilePosition,
) -> Option<lsp_ext::CommandLinkGroup> {
    if snap.config.hover_actions().references && snap.config.client_commands().show_reference {
        if let Some(ref_search_res) = snap.analysis.find_all_refs(*position, None).unwrap_or(None) {
            let uri = to_proto::url(snap, position.file_id);
            let line_index = snap.file_line_index(position.file_id).ok()?;
            let position = to_proto::position(&line_index, position.offset);
            let locations: Vec<_> = ref_search_res
                .into_iter()
                .flat_map(|res| res.references)
                .flat_map(|(file_id, ranges)| {
                    ranges.into_iter().filter_map(move |(range, _)| {
                        to_proto::location(snap, FileRange { file_id, range }).ok()
                    })
                })
                .collect();
            let title = to_proto::reference_title(locations.len());
            let command = to_proto::command::show_references(title, &uri, position, locations);

            return Some(lsp_ext::CommandLinkGroup {
                commands: vec![to_command_link(command, "Go to references".into())],
                ..Default::default()
            });
        }
    }
    None
}

fn runnable_action_links(
    snap: &GlobalStateSnapshot,
    runnable: Runnable,
) -> Option<lsp_ext::CommandLinkGroup> {
    let hover_actions_config = snap.config.hover_actions();
    if !hover_actions_config.runnable() {
        return None;
    }

    let cargo_spec = CargoTargetSpec::for_file(snap, runnable.nav.file_id).ok()?;
    if should_skip_target(&runnable, cargo_spec.as_ref()) {
        return None;
    }

    let client_commands_config = snap.config.client_commands();
    if !(client_commands_config.run_single || client_commands_config.debug_single) {
        return None;
    }

    let title = runnable.title();
    let r = to_proto::runnable(snap, runnable).ok()?;

    let mut group = lsp_ext::CommandLinkGroup::default();

    if hover_actions_config.run && client_commands_config.run_single {
        let run_command = to_proto::command::run_single(&r, &title);
        group.commands.push(to_command_link(run_command, r.label.clone()));
    }

    if hover_actions_config.debug && client_commands_config.debug_single {
        let dbg_command = to_proto::command::debug_single(&r);
        group.commands.push(to_command_link(dbg_command, r.label));
    }

    Some(group)
}

fn goto_type_action_links(
    snap: &GlobalStateSnapshot,
    nav_targets: &[HoverGotoTypeData],
) -> Option<lsp_ext::CommandLinkGroup> {
    if !snap.config.hover_actions().goto_type_def
        || nav_targets.is_empty()
        || !snap.config.client_commands().goto_location
    {
        return None;
    }

    Some(lsp_ext::CommandLinkGroup {
        title: Some("Go to ".into()),
        commands: nav_targets
            .iter()
            .filter_map(|it| {
                to_proto::command::goto_location(snap, &it.nav)
                    .map(|cmd| to_command_link(cmd, it.mod_path.clone()))
            })
            .collect(),
    })
}

fn prepare_hover_actions(
    snap: &GlobalStateSnapshot,
    actions: &[HoverAction],
) -> Vec<lsp_ext::CommandLinkGroup> {
    actions
        .iter()
        .filter_map(|it| match it {
            HoverAction::Implementation(position) => show_impl_command_link(snap, position),
            HoverAction::Reference(position) => show_ref_command_link(snap, position),
            HoverAction::Runnable(r) => runnable_action_links(snap, r.clone()),
            HoverAction::GoToType(targets) => goto_type_action_links(snap, targets),
        })
        .collect()
}

fn should_skip_target(runnable: &Runnable, cargo_spec: Option<&CargoTargetSpec>) -> bool {
    match runnable.kind {
        RunnableKind::Bin => {
            // Do not suggest binary run on other target than binary
            match &cargo_spec {
                Some(spec) => !matches!(
                    spec.target_kind,
                    TargetKind::Bin | TargetKind::Example | TargetKind::Test
                ),
                None => true,
            }
        }
        _ => false,
    }
}

fn run_rustfmt(
    snap: &GlobalStateSnapshot,
    text_document: TextDocumentIdentifier,
    range: Option<lsp_types::Range>,
) -> Result<Option<Vec<lsp_types::TextEdit>>> {
    let file_id = from_proto::file_id(snap, &text_document.uri)?;
    let file = snap.analysis.file_text(file_id)?;
    let crate_ids = snap.analysis.crate_for(file_id)?;

    let line_index = snap.file_line_index(file_id)?;

    let mut command = match snap.config.rustfmt() {
        RustfmtConfig::Rustfmt { extra_args, enable_range_formatting } => {
            let mut cmd = process::Command::new(toolchain::rustfmt());
            cmd.args(extra_args);
            // try to chdir to the file so we can respect `rustfmt.toml`
            // FIXME: use `rustfmt --config-path` once
            // https://github.com/rust-lang/rustfmt/issues/4660 gets fixed
            match text_document.uri.to_file_path() {
                Ok(mut path) => {
                    // pop off file name
                    if path.pop() && path.is_dir() {
                        cmd.current_dir(path);
                    }
                }
                Err(_) => {
                    tracing::error!(
                        "Unable to get file path for {}, rustfmt.toml might be ignored",
                        text_document.uri
                    );
                }
            }
            if let Some(&crate_id) = crate_ids.first() {
                // Assume all crates are in the same edition
                let edition = snap.analysis.crate_edition(crate_id)?;
                cmd.arg("--edition");
                cmd.arg(edition.to_string());
            }

            if let Some(range) = range {
                if !enable_range_formatting {
                    return Err(LspError::new(
                        ErrorCode::InvalidRequest as i32,
                        String::from(
                            "rustfmt range formatting is unstable. \
                            Opt-in by using a nightly build of rustfmt and setting \
                            `rustfmt.rangeFormatting.enable` to true in your LSP configuration",
                        ),
                    )
                    .into());
                }

                let frange = from_proto::file_range(snap, text_document, range)?;
                let start_line = line_index.index.line_col(frange.range.start()).line;
                let end_line = line_index.index.line_col(frange.range.end()).line;

                cmd.arg("--unstable-features");
                cmd.arg("--file-lines");
                cmd.arg(
                    json!([{
                        "file": "stdin",
                        "range": [start_line, end_line]
                    }])
                    .to_string(),
                );
            }

            cmd
        }
        RustfmtConfig::CustomCommand { command, args } => {
            let mut cmd = process::Command::new(command);
            cmd.args(args);
            cmd
        }
    };

    let mut rustfmt = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context(format!("Failed to spawn {:?}", command))?;

    rustfmt.stdin.as_mut().unwrap().write_all(file.as_bytes())?;

    let output = rustfmt.wait_with_output()?;
    let captured_stdout = String::from_utf8(output.stdout)?;
    let captured_stderr = String::from_utf8(output.stderr).unwrap_or_default();

    if !output.status.success() {
        let rustfmt_not_installed =
            captured_stderr.contains("not installed") || captured_stderr.contains("not available");

        return match output.status.code() {
            Some(1) if !rustfmt_not_installed => {
                // While `rustfmt` doesn't have a specific exit code for parse errors this is the
                // likely cause exiting with 1. Most Language Servers swallow parse errors on
                // formatting because otherwise an error is surfaced to the user on top of the
                // syntax error diagnostics they're already receiving. This is especially jarring
                // if they have format on save enabled.
                tracing::warn!(
                    ?command,
                    %captured_stderr,
                    "rustfmt exited with status 1"
                );
                Ok(None)
            }
            _ => {
                // Something else happened - e.g. `rustfmt` is missing or caught a signal
                Err(LspError::new(
                    -32900,
                    format!(
                        r#"rustfmt exited with:
                           Status: {}
                           stdout: {}
                           stderr: {}"#,
                        output.status, captured_stdout, captured_stderr,
                    ),
                )
                .into())
            }
        };
    }

    let (new_text, new_line_endings) = LineEndings::normalize(captured_stdout);

    if line_index.endings != new_line_endings {
        // If line endings are different, send the entire file.
        // Diffing would not work here, as the line endings might be the only
        // difference.
        Ok(Some(to_proto::text_edit_vec(
            &line_index,
            TextEdit::replace(TextRange::up_to(TextSize::of(&*file)), new_text),
        )))
    } else if *file == new_text {
        // The document is already formatted correctly -- no edits needed.
        Ok(None)
    } else {
        Ok(Some(to_proto::text_edit_vec(&line_index, diff(&file, &new_text))))
    }
}
