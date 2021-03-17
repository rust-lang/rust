//! This module is responsible for implementing handlers for Language Server
//! Protocol. The majority of requests are fulfilled by calling into the
//! `ide` crate.

use std::{
    io::Write as _,
    process::{self, Stdio},
};

use ide::{
    AnnotationConfig, FileId, FilePosition, FileRange, HoverAction, HoverGotoTypeData, Query,
    RangeInfo, Runnable, RunnableKind, SearchScope, SourceChange, TextEdit,
};
use ide_db::SymbolKind;
use itertools::Itertools;
use lsp_server::ErrorCode;
use lsp_types::{
    CallHierarchyIncomingCall, CallHierarchyIncomingCallsParams, CallHierarchyItem,
    CallHierarchyOutgoingCall, CallHierarchyOutgoingCallsParams, CallHierarchyPrepareParams,
    CodeActionKind, CodeLens, CompletionItem, Diagnostic, DiagnosticTag, DocumentFormattingParams,
    DocumentHighlight, FoldingRange, FoldingRangeParams, HoverContents, Location, NumberOrString,
    Position, PrepareRenameResponse, Range, RenameParams, SemanticTokensDeltaParams,
    SemanticTokensFullDeltaResult, SemanticTokensParams, SemanticTokensRangeParams,
    SemanticTokensRangeResult, SemanticTokensResult, SymbolInformation, SymbolTag,
    TextDocumentIdentifier, TextDocumentPositionParams, Url, WorkspaceEdit,
};
use project_model::TargetKind;
use serde::{Deserialize, Serialize};
use serde_json::to_value;
use stdx::{format_to, split_once};
use syntax::{algo, ast, AstNode, TextRange, TextSize};

use crate::{
    cargo_target_spec::CargoTargetSpec,
    config::RustfmtConfig,
    diff::diff,
    from_proto,
    global_state::{GlobalState, GlobalStateSnapshot},
    line_index::{LineEndings, LineIndex},
    lsp_ext::{self, InlayHint, InlayHintsParams},
    lsp_utils::all_edits_are_disjoint,
    to_proto, LspError, Result,
};

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
        buf.push_str("no workspaces\n")
    } else {
        buf.push_str("workspaces:\n");
        for w in snap.workspaces.iter() {
            format_to!(buf, "{} packages loaded\n", w.n_packages());
        }
    }
    buf.push_str("\nanalysis:\n");
    buf.push_str(
        &snap
            .analysis
            .status(file_id)
            .unwrap_or_else(|_| "Analysis retrieval was cancelled".to_owned()),
    );
    format_to!(buf, "\n\nrequests:\n");
    let requests = snap.latest_requests.read();
    for (is_last, r) in requests.iter() {
        let mark = if is_last { "*" } else { " " };
        format_to!(buf, "{}{:4} {:<36}{}ms\n", mark, r.id, r.method, r.duration.as_millis());
    }
    Ok(buf)
}

pub(crate) fn handle_memory_usage(state: &mut GlobalState, _: ()) -> Result<String> {
    let _p = profile::span("handle_memory_usage");
    let mem = state.analysis_host.per_query_memory_usage();

    let mut out = String::new();
    for (name, bytes) in mem {
        format_to!(out, "{:>8} {}\n", bytes, name);
    }
    Ok(out)
}

pub(crate) fn handle_syntax_tree(
    snap: GlobalStateSnapshot,
    params: lsp_ext::SyntaxTreeParams,
) -> Result<String> {
    let _p = profile::span("handle_syntax_tree");
    let id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.file_line_index(id)?;
    let text_range = params.range.map(|r| from_proto::text_range(&line_index, r));
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

pub(crate) fn handle_expand_macro(
    snap: GlobalStateSnapshot,
    params: lsp_ext::ExpandMacroParams,
) -> Result<Option<lsp_ext::ExpandedMacro>> {
    let _p = profile::span("handle_expand_macro");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;
    let offset = from_proto::offset(&line_index, params.position);

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
            let offset = from_proto::offset(&line_index, position);
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
    let res = params
        .positions
        .into_iter()
        .map(|position| {
            let offset = from_proto::offset(&line_index, position);
            let offset = match snap.analysis.matching_brace(FilePosition { file_id, offset }) {
                Ok(Some(matching_brace_offset)) => matching_brace_offset,
                Err(_) | Ok(None) => offset,
            };
            to_proto::position(&line_index, offset)
        })
        .collect();
    Ok(res)
}

pub(crate) fn handle_join_lines(
    snap: GlobalStateSnapshot,
    params: lsp_ext::JoinLinesParams,
) -> Result<Vec<lsp_types::TextEdit>> {
    let _p = profile::span("handle_join_lines");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;
    let mut res = TextEdit::default();
    for range in params.ranges {
        let range = from_proto::text_range(&line_index, range);
        let edit = snap.analysis.join_lines(FileRange { file_id, range })?;
        match res.union(edit) {
            Ok(()) => (),
            Err(_edit) => {
                // just ignore overlapping edits
            }
        }
    }
    let res = to_proto::text_edit_vec(&line_index, res);
    Ok(res)
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

// Don't forget to add new trigger characters to `ServerCapabilities` in `caps.rs`.
pub(crate) fn handle_on_type_formatting(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentOnTypeFormattingParams,
) -> Result<Option<Vec<lsp_types::TextEdit>>> {
    let _p = profile::span("handle_on_type_formatting");
    let mut position = from_proto::file_position(&snap, params.text_document_position)?;
    let line_index = snap.file_line_index(position.file_id)?;

    // in `ide`, the `on_type` invariant is that
    // `text.char_at(position) == typed_char`.
    position.offset -= TextSize::of('.');
    let char_typed = params.ch.chars().next().unwrap_or('\0');
    assert!({
        let text = snap.analysis.file_text(position.file_id)?;
        text[usize::from(position.offset)..].starts_with(char_typed)
    });

    // We have an assist that inserts ` ` after typing `->` in `fn foo() ->{`,
    // but it requires precise cursor positioning to work, and one can't
    // position the cursor with on_type formatting. So, let's just toggle this
    // feature off here, hoping that we'll enable it one day, 😿.
    if char_typed == '>' {
        return Ok(None);
    }

    let edit = snap.analysis.on_char_typed(position, char_typed)?;
    let edit = match edit {
        Some(it) => it,
        None => return Ok(None),
    };

    // This should be a single-file edit
    let (_, edit) = edit.source_file_edits.into_iter().next().unwrap();

    let change = to_proto::text_edit_vec(&line_index, edit);
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
            tags.push(SymbolTag::Deprecated)
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
        match symbol.deprecated {
            Some(true) => tags.push(SymbolTag::Deprecated),
            _ => {}
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
    params: lsp_types::WorkspaceSymbolParams,
) -> Result<Option<Vec<SymbolInformation>>> {
    let _p = profile::span("handle_workspace_symbol");
    let all_symbols = params.query.contains('#');
    let libs = params.query.contains('*');
    let query = {
        let query: String = params.query.chars().filter(|&c| c != '#' && c != '*').collect();
        let mut q = Query::new(query);
        if !all_symbols {
            q.only_types();
        }
        if libs {
            q.libs();
        }
        q.limit(128);
        q
    };
    let mut res = exec_query(&snap, query)?;
    if res.is_empty() && !all_symbols {
        let mut query = Query::new(params.query);
        query.limit(128);
        res = exec_query(&snap, query)?;
    }

    return Ok(Some(res));

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
                    .unwrap_or(lsp_types::SymbolKind::Variable),
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
    source_change.extend(source_changes.map(|it| it.source_file_edits).flatten());
    let workspace_edit = to_proto::workspace_edit(&snap, source_change)?;
    Ok(Some(workspace_edit))
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
    let offset = params.position.map(|it| from_proto::offset(&line_index, it));
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
        if let Some(offset) = offset {
            if !runnable.nav.full_range.contains_inclusive(offset) {
                continue;
            }
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
            for &cmd in ["check", "test"].iter() {
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
    Ok(res)
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
    let completion_triggered_after_single_colon = {
        let mut res = false;
        if let Some(ctx) = params.context {
            if ctx.trigger_character.unwrap_or_default() == ":" {
                let source_file = snap.analysis.parse(position.file_id)?;
                let syntax = source_file.syntax();
                let text = syntax.text();
                if let Some(next_char) = text.char_at(position.offset) {
                    let diff = TextSize::of(next_char) + TextSize::of(':');
                    let prev_char = position.offset - diff;
                    if text.char_at(prev_char) != Some(':') {
                        res = true;
                    }
                }
            }
        }
        res
    };
    if completion_triggered_after_single_colon {
        return Ok(None);
    }

    let completion_config = &snap.config.completion();
    let items = match snap.analysis.completions(completion_config, position)? {
        None => return Ok(None),
        Some(items) => items,
    };
    let line_index = snap.file_line_index(position.file_id)?;

    let items: Vec<CompletionItem> = items
        .into_iter()
        .flat_map(|item| {
            let mut new_completion_items = to_proto::completion_item(&line_index, item.clone());

            if completion_config.enable_imports_on_the_fly {
                for new_item in &mut new_completion_items {
                    fill_resolve_data(&mut new_item.data, &item, &text_document_position);
                }
            }

            new_completion_items
        })
        .collect();

    let completion_list = lsp_types::CompletionList { is_incomplete: true, items };
    Ok(Some(completion_list.into()))
}

pub(crate) fn handle_completion_resolve(
    snap: GlobalStateSnapshot,
    mut original_completion: CompletionItem,
) -> Result<CompletionItem> {
    let _p = profile::span("handle_completion_resolve");

    if !all_edits_are_disjoint(&original_completion, &[]) {
        return Err(LspError::new(
            ErrorCode::InvalidParams as i32,
            "Received a completion with overlapping edits, this is not LSP-compliant".into(),
        )
        .into());
    }

    let resolve_data = match original_completion
        .data
        .take()
        .map(|data| serde_json::from_value::<CompletionResolveData>(data))
        .transpose()?
    {
        Some(data) => data,
        None => return Ok(original_completion),
    };

    let file_id = from_proto::file_id(&snap, &resolve_data.position.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;
    let offset = from_proto::offset(&line_index, resolve_data.position.position);

    let additional_edits = snap
        .analysis
        .resolve_completion_edits(
            &snap.config.completion(),
            FilePosition { file_id, offset },
            &resolve_data.full_import_path,
            resolve_data.imported_name,
        )?
        .into_iter()
        .flat_map(|edit| edit.into_iter().map(|indel| to_proto::text_edit(&line_index, indel)))
        .collect_vec();

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
    let call_info = match snap.analysis.call_info(position)? {
        Some(it) => it,
        None => return Ok(None),
    };
    let concise = !snap.config.call_info_full();
    let res =
        to_proto::signature_help(call_info, concise, snap.config.signature_help_label_offsets());
    Ok(Some(res))
}

pub(crate) fn handle_hover(
    snap: GlobalStateSnapshot,
    params: lsp_types::HoverParams,
) -> Result<Option<lsp_ext::Hover>> {
    let _p = profile::span("handle_hover");
    let position = from_proto::file_position(&snap, params.text_document_position_params)?;
    let hover_config = snap.config.hover();
    let info =
        match snap.analysis.hover(position, hover_config.links_in_hover, hover_config.markdown)? {
            None => return Ok(None),
            Some(info) => info,
        };
    let line_index = snap.file_line_index(position.file_id)?;
    let range = to_proto::range(&line_index, info.range);
    let hover = lsp_ext::Hover {
        hover: lsp_types::Hover {
            contents: HoverContents::Markup(to_proto::markup_content(info.info.markup)),
            range: Some(range),
        },
        actions: prepare_hover_actions(&snap, &info.info.actions),
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

    let decl = if params.context.include_declaration {
        refs.declaration.map(|decl| FileRange {
            file_id: decl.nav.file_id,
            range: decl.nav.focus_or_full_range(),
        })
    } else {
        None
    };
    let locations = refs
        .references
        .into_iter()
        .flat_map(|(file_id, refs)| {
            refs.into_iter().map(move |(range, _)| FileRange { file_id, range })
        })
        .chain(decl)
        .filter_map(|frange| to_proto::location(&snap, frange).ok())
        .collect();

    Ok(Some(locations))
}

pub(crate) fn handle_formatting(
    snap: GlobalStateSnapshot,
    params: DocumentFormattingParams,
) -> Result<Option<Vec<lsp_types::TextEdit>>> {
    let _p = profile::span("handle_formatting");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let file = snap.analysis.file_text(file_id)?;
    let crate_ids = snap.analysis.crate_for(file_id)?;

    let line_index = snap.file_line_index(file_id)?;

    let mut rustfmt = match snap.config.rustfmt() {
        RustfmtConfig::Rustfmt { extra_args } => {
            let mut cmd = process::Command::new(toolchain::rustfmt());
            cmd.args(extra_args);
            // try to chdir to the file so we can respect `rustfmt.toml`
            // FIXME: use `rustfmt --config-path` once
            // https://github.com/rust-lang/rustfmt/issues/4660 gets fixed
            match params.text_document.uri.to_file_path() {
                Ok(mut path) => {
                    // pop off file name
                    if path.pop() && path.is_dir() {
                        cmd.current_dir(path);
                    }
                }
                Err(_) => {
                    log::error!(
                        "Unable to get file path for {}, rustfmt.toml might be ignored",
                        params.text_document.uri
                    );
                }
            }
            if let Some(&crate_id) = crate_ids.first() {
                // Assume all crates are in the same edition
                let edition = snap.analysis.crate_edition(crate_id)?;
                cmd.arg("--edition");
                cmd.arg(edition.to_string());
            }
            cmd
        }
        RustfmtConfig::CustomCommand { command, args } => {
            let mut cmd = process::Command::new(command);
            cmd.args(args);
            cmd
        }
    };

    let mut rustfmt =
        rustfmt.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;

    rustfmt.stdin.as_mut().unwrap().write_all(file.as_bytes())?;

    let output = rustfmt.wait_with_output()?;
    let captured_stdout = String::from_utf8(output.stdout)?;
    let captured_stderr = String::from_utf8(output.stderr).unwrap_or_default();

    if !output.status.success() {
        match output.status.code() {
            Some(1) if !captured_stderr.contains("not installed") => {
                // While `rustfmt` doesn't have a specific exit code for parse errors this is the
                // likely cause exiting with 1. Most Language Servers swallow parse errors on
                // formatting because otherwise an error is surfaced to the user on top of the
                // syntax error diagnostics they're already receiving. This is especially jarring
                // if they have format on save enabled.
                log::info!("rustfmt exited with status 1, assuming parse error and ignoring");
                return Ok(None);
            }
            _ => {
                // Something else happened - e.g. `rustfmt` is missing or caught a signal
                return Err(LspError::new(
                    -32900,
                    format!(
                        r#"rustfmt exited with:
                           Status: {}
                           stdout: {}
                           stderr: {}"#,
                        output.status, captured_stdout, captured_stderr,
                    ),
                )
                .into());
            }
        }
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

pub(crate) fn handle_code_action(
    snap: GlobalStateSnapshot,
    params: lsp_types::CodeActionParams,
) -> Result<Option<Vec<lsp_ext::CodeAction>>> {
    let _p = profile::span("handle_code_action");
    // We intentionally don't support command-based actions, as those either
    // requires custom client-code anyway, or requires server-initiated edits.
    // Server initiated edits break causality, so we avoid those as well.
    if !snap.config.code_action_literals() {
        return Ok(None);
    }

    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;
    let range = from_proto::text_range(&line_index, params.range);
    let frange = FileRange { file_id, range };

    let mut assists_config = snap.config.assist();
    assists_config.allowed = params
        .clone()
        .context
        .only
        .map(|it| it.into_iter().filter_map(from_proto::assist_kind).collect());

    let mut res: Vec<lsp_ext::CodeAction> = Vec::new();

    let include_quick_fixes = match &params.context.only {
        Some(v) => v.iter().any(|it| {
            it == &lsp_types::CodeActionKind::EMPTY || it == &lsp_types::CodeActionKind::QUICKFIX
        }),
        None => true,
    };
    if include_quick_fixes {
        add_quick_fixes(&snap, frange, &line_index, &mut res)?;
    }

    if snap.config.code_action_resolve() {
        for (index, assist) in
            snap.analysis.assists(&assists_config, false, frange)?.into_iter().enumerate()
        {
            res.push(to_proto::unresolved_code_action(&snap, params.clone(), assist, index)?);
        }
    } else {
        for assist in snap.analysis.assists(&assists_config, true, frange)?.into_iter() {
            res.push(to_proto::resolved_code_action(&snap, assist)?);
        }
    }

    Ok(Some(res))
}

fn add_quick_fixes(
    snap: &GlobalStateSnapshot,
    frange: FileRange,
    line_index: &LineIndex,
    acc: &mut Vec<lsp_ext::CodeAction>,
) -> Result<()> {
    let diagnostics = snap.analysis.diagnostics(&snap.config.diagnostics(), frange.file_id)?;

    for fix in diagnostics
        .into_iter()
        .filter_map(|d| d.fix)
        .filter(|fix| fix.fix_trigger_range.intersect(frange.range).is_some())
    {
        let edit = to_proto::snippet_workspace_edit(&snap, fix.source_change)?;
        let action = lsp_ext::CodeAction {
            title: fix.label.to_string(),
            group: None,
            kind: Some(CodeActionKind::QUICKFIX),
            edit: Some(edit),
            is_preferred: Some(false),
            data: None,
        };
        acc.push(action);
    }

    for fix in snap.check_fixes.get(&frange.file_id).into_iter().flatten() {
        let fix_range = from_proto::text_range(&line_index, fix.range);
        if fix_range.intersect(frange.range).is_some() {
            acc.push(fix.action.clone());
        }
    }
    Ok(())
}

pub(crate) fn handle_code_action_resolve(
    snap: GlobalStateSnapshot,
    mut code_action: lsp_ext::CodeAction,
) -> Result<lsp_ext::CodeAction> {
    let _p = profile::span("handle_code_action_resolve");
    let params = match code_action.data.take() {
        Some(it) => it,
        None => Err("can't resolve code action without data")?,
    };

    let file_id = from_proto::file_id(&snap, &params.code_action_params.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;
    let range = from_proto::text_range(&line_index, params.code_action_params.range);
    let frange = FileRange { file_id, range };

    let mut assists_config = snap.config.assist();
    assists_config.allowed = params
        .code_action_params
        .context
        .only
        .map(|it| it.into_iter().filter_map(from_proto::assist_kind).collect());

    let assists = snap.analysis.assists(&assists_config, true, frange)?;
    let (id, index) = split_once(&params.id, ':').unwrap();
    let index = index.parse::<usize>().unwrap();
    let assist = &assists[index];
    assert!(assist.id.0 == id);
    let edit = to_proto::resolved_code_action(&snap, assist.clone())?.edit;
    code_action.edit = edit;
    Ok(code_action)
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

    let lenses = snap
        .analysis
        .annotations(
            file_id,
            AnnotationConfig {
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
                annotate_references: lens_config.refs,
                annotate_method_references: lens_config.method_refs,
                run: lens_config.run,
                debug: lens_config.debug,
            },
        )?
        .into_iter()
        .map(|annotation| to_proto::code_lens(&snap, annotation).unwrap())
        .collect();

    Ok(Some(lenses))
}

pub(crate) fn handle_code_lens_resolve(
    snap: GlobalStateSnapshot,
    code_lens: CodeLens,
) -> Result<CodeLens> {
    let annotation = from_proto::annotation(&snap, code_lens)?;

    to_proto::code_lens(&snap, snap.analysis.resolve_annotation(annotation)?)
}

pub(crate) fn handle_document_highlight(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentHighlightParams,
) -> Result<Option<Vec<DocumentHighlight>>> {
    let _p = profile::span("handle_document_highlight");
    let position = from_proto::file_position(&snap, params.text_document_position_params)?;
    let line_index = snap.file_line_index(position.file_id)?;

    let refs = match snap
        .analysis
        .find_all_refs(position, Some(SearchScope::single_file(position.file_id)))?
    {
        None => return Ok(None),
        Some(refs) => refs,
    };

    let decl = refs.declaration.filter(|decl| decl.nav.file_id == position.file_id).map(|decl| {
        DocumentHighlight {
            range: to_proto::range(&line_index, decl.nav.focus_or_full_range()),
            kind: decl.access.map(to_proto::document_highlight_kind),
        }
    });

    let file_refs = refs.references.get(&position.file_id).map_or(&[][..], Vec::as_slice);
    let mut res = Vec::with_capacity(file_refs.len() + 1);
    res.extend(decl);
    res.extend(file_refs.iter().map(|&(range, access)| DocumentHighlight {
        range: to_proto::range(&line_index, range),
        kind: access.map(to_proto::document_highlight_kind),
    }));
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
        .diagnostics(&snap.config.diagnostics(), file_id)?
        .into_iter()
        .map(|d| Diagnostic {
            range: to_proto::range(&line_index, d.range),
            severity: Some(to_proto::diagnostic_severity(d.severity)),
            code: d.code.map(|d| d.as_str().to_owned()).map(NumberOrString::String),
            code_description: d.code.and_then(|code| {
                lsp_types::Url::parse(&format!(
                    "https://rust-analyzer.github.io/manual.html#{}",
                    code.as_str()
                ))
                .ok()
                .map(|href| lsp_types::CodeDescription { href })
            }),
            source: Some("rust-analyzer".to_string()),
            message: d.message,
            related_information: None,
            tags: if d.unused { Some(vec![DiagnosticTag::Unnecessary]) } else { None },
            data: None,
        })
        .collect();
    Ok(diagnostics)
}

pub(crate) fn handle_inlay_hints(
    snap: GlobalStateSnapshot,
    params: InlayHintsParams,
) -> Result<Vec<InlayHint>> {
    let _p = profile::span("handle_inlay_hints");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;
    Ok(snap
        .analysis
        .inlay_hints(file_id, &snap.config.inlay_hints())?
        .into_iter()
        .map(|it| to_proto::inlay_hint(&line_index, it))
        .collect())
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

    let highlights = snap.analysis.highlight(file_id)?;
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

    let highlights = snap.analysis.highlight(file_id)?;

    let semantic_tokens = to_proto::semantic_tokens(&text, &line_index, highlights);

    let mut cache = snap.semantic_tokens_cache.lock();
    let cached_tokens = cache.entry(params.text_document.uri).or_default();

    if let Some(prev_id) = &cached_tokens.result_id {
        if *prev_id == params.previous_result_id {
            let delta = to_proto::semantic_token_delta(&cached_tokens, &semantic_tokens);
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

    let highlights = snap.analysis.highlight_range(frange)?;
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

fn to_command_link(command: lsp_types::Command, tooltip: String) -> lsp_ext::CommandLink {
    lsp_ext::CommandLink { tooltip: Some(tooltip), command }
}

fn show_impl_command_link(
    snap: &GlobalStateSnapshot,
    position: &FilePosition,
) -> Option<lsp_ext::CommandLinkGroup> {
    if snap.config.hover().implementations {
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

fn runnable_action_links(
    snap: &GlobalStateSnapshot,
    runnable: Runnable,
) -> Option<lsp_ext::CommandLinkGroup> {
    let cargo_spec = CargoTargetSpec::for_file(&snap, runnable.nav.file_id).ok()?;
    let hover_config = snap.config.hover();
    if !hover_config.runnable() || should_skip_target(&runnable, cargo_spec.as_ref()) {
        return None;
    }

    let action: &'static _ = runnable.action();
    to_proto::runnable(snap, runnable).ok().map(|r| {
        let mut group = lsp_ext::CommandLinkGroup::default();

        if hover_config.run {
            let run_command = to_proto::command::run_single(&r, action.run_title);
            group.commands.push(to_command_link(run_command, r.label.clone()));
        }

        if hover_config.debug {
            let dbg_command = to_proto::command::debug_single(&r);
            group.commands.push(to_command_link(dbg_command, r.label));
        }

        group
    })
}

fn goto_type_action_links(
    snap: &GlobalStateSnapshot,
    nav_targets: &[HoverGotoTypeData],
) -> Option<lsp_ext::CommandLinkGroup> {
    if !snap.config.hover().goto_type_def || nav_targets.is_empty() {
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
    if snap.config.hover().none() || !snap.config.hover_actions() {
        return Vec::new();
    }

    actions
        .iter()
        .filter_map(|it| match it {
            HoverAction::Implementation(position) => show_impl_command_link(snap, position),
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

#[derive(Debug, Serialize, Deserialize)]
struct CompletionResolveData {
    position: lsp_types::TextDocumentPositionParams,
    full_import_path: String,
    imported_name: String,
}

fn fill_resolve_data(
    resolve_data: &mut Option<serde_json::Value>,
    item: &ide::CompletionItem,
    position: &TextDocumentPositionParams,
) -> Option<()> {
    let import_edit = item.import_to_add()?;
    let import_path = &import_edit.import.import_path;

    *resolve_data = Some(
        to_value(CompletionResolveData {
            position: position.to_owned(),
            full_import_path: import_path.to_string(),
            imported_name: import_path.segments().last()?.to_string(),
        })
        .unwrap(),
    );
    Some(())
}
