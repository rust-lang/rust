//! This module is responsible for implementing handlers for Language Server
//! Protocol. The majority of requests are fulfilled by calling into the
//! `ra_ide` crate.

use std::{
    io::Write as _,
    process::{self, Stdio},
};

use lsp_server::ErrorCode;
use lsp_types::{
    CallHierarchyIncomingCall, CallHierarchyIncomingCallsParams, CallHierarchyItem,
    CallHierarchyOutgoingCall, CallHierarchyOutgoingCallsParams, CallHierarchyPrepareParams,
    CodeLens, Command, CompletionItem, Diagnostic, DocumentFormattingParams, DocumentHighlight,
    DocumentSymbol, FoldingRange, FoldingRangeParams, HoverContents, Location, MarkupContent,
    MarkupKind, Position, PrepareRenameResponse, Range, RenameParams, SemanticTokensParams,
    SemanticTokensRangeParams, SemanticTokensRangeResult, SemanticTokensResult, SymbolInformation,
    TextDocumentIdentifier, Url, WorkspaceEdit,
};
use ra_ide::{
    FileId, FilePosition, FileRange, HoverAction, HoverGotoTypeData, NavigationTarget, Query,
    RangeInfo, Runnable, RunnableKind, SearchScope, TextEdit,
};
use ra_prof::profile;
use ra_project_model::TargetKind;
use ra_syntax::{algo, ast, AstNode, SyntaxKind, TextRange, TextSize};
use serde::{Deserialize, Serialize};
use serde_json::to_value;
use stdx::{format_to, split_delim};

use crate::{
    cargo_target_spec::CargoTargetSpec,
    config::RustfmtConfig,
    from_json, from_proto,
    global_state::GlobalStateSnapshot,
    lsp_ext::{self, InlayHint, InlayHintsParams},
    to_proto, LspError, Result,
};

pub(crate) fn handle_analyzer_status(snap: GlobalStateSnapshot, _: ()) -> Result<String> {
    let _p = profile("handle_analyzer_status");

    let mut buf = String::new();
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
        &snap.analysis.status().unwrap_or_else(|_| "Analysis retrieval was cancelled".to_owned()),
    );
    format_to!(buf, "\n\nrequests:\n");
    let requests = snap.latest_requests.read();
    for (is_last, r) in requests.iter() {
        let mark = if is_last { "*" } else { " " };
        format_to!(buf, "{}{:4} {:<36}{}ms\n", mark, r.id, r.method, r.duration.as_millis());
    }
    Ok(buf)
}

pub(crate) fn handle_syntax_tree(
    snap: GlobalStateSnapshot,
    params: lsp_ext::SyntaxTreeParams,
) -> Result<String> {
    let _p = profile("handle_syntax_tree");
    let id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.analysis.file_line_index(id)?;
    let text_range = params.range.map(|r| from_proto::text_range(&line_index, r));
    let res = snap.analysis.syntax_tree(id, text_range)?;
    Ok(res)
}

pub(crate) fn handle_expand_macro(
    snap: GlobalStateSnapshot,
    params: lsp_ext::ExpandMacroParams,
) -> Result<Option<lsp_ext::ExpandedMacro>> {
    let _p = profile("handle_expand_macro");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.analysis.file_line_index(file_id)?;
    let offset = from_proto::offset(&line_index, params.position);

    let res = snap.analysis.expand_macro(FilePosition { file_id, offset })?;
    Ok(res.map(|it| lsp_ext::ExpandedMacro { name: it.name, expansion: it.expansion }))
}

pub(crate) fn handle_selection_range(
    snap: GlobalStateSnapshot,
    params: lsp_types::SelectionRangeParams,
) -> Result<Option<Vec<lsp_types::SelectionRange>>> {
    let _p = profile("handle_selection_range");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.analysis.file_line_index(file_id)?;
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
    let _p = profile("handle_matching_brace");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.analysis.file_line_index(file_id)?;
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
    let _p = profile("handle_join_lines");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.analysis.file_line_index(file_id)?;
    let line_endings = snap.file_line_endings(file_id);
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
    let res = to_proto::text_edit_vec(&line_index, line_endings, res);
    Ok(res)
}

pub(crate) fn handle_on_enter(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> Result<Option<Vec<lsp_ext::SnippetTextEdit>>> {
    let _p = profile("handle_on_enter");
    let position = from_proto::file_position(&snap, params)?;
    let edit = match snap.analysis.on_enter(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let line_index = snap.analysis.file_line_index(position.file_id)?;
    let line_endings = snap.file_line_endings(position.file_id);
    let edit = to_proto::snippet_text_edit_vec(&line_index, line_endings, true, edit);
    Ok(Some(edit))
}

// Don't forget to add new trigger characters to `ServerCapabilities` in `caps.rs`.
pub(crate) fn handle_on_type_formatting(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentOnTypeFormattingParams,
) -> Result<Option<Vec<lsp_types::TextEdit>>> {
    let _p = profile("handle_on_type_formatting");
    let mut position = from_proto::file_position(&snap, params.text_document_position)?;
    let line_index = snap.analysis.file_line_index(position.file_id)?;
    let line_endings = snap.file_line_endings(position.file_id);

    // in `ra_ide`, the `on_type` invariant is that
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
    let mut edit = match edit {
        Some(it) => it,
        None => return Ok(None),
    };

    // This should be a single-file edit
    let edit = edit.source_file_edits.pop().unwrap();

    let change = to_proto::text_edit_vec(&line_index, line_endings, edit.edit);
    Ok(Some(change))
}

pub(crate) fn handle_document_symbol(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentSymbolParams,
) -> Result<Option<lsp_types::DocumentSymbolResponse>> {
    let _p = profile("handle_document_symbol");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.analysis.file_line_index(file_id)?;

    let mut parents: Vec<(DocumentSymbol, Option<usize>)> = Vec::new();

    for symbol in snap.analysis.file_structure(file_id)? {
        let doc_symbol = DocumentSymbol {
            name: symbol.label,
            detail: symbol.detail,
            kind: to_proto::symbol_kind(symbol.kind),
            deprecated: Some(symbol.deprecated),
            range: to_proto::range(&line_index, symbol.node_range),
            selection_range: to_proto::range(&line_index, symbol.navigation_range),
            children: None,
        };
        parents.push((doc_symbol, symbol.parent));
    }
    let mut document_symbols = Vec::new();
    while let Some((node, parent)) = parents.pop() {
        match parent {
            None => document_symbols.push(node),
            Some(i) => {
                let children = &mut parents[i].0.children;
                if children.is_none() {
                    *children = Some(Vec::new());
                }
                children.as_mut().unwrap().push(node);
            }
        }
    }

    let res = if snap.config.client_caps.hierarchical_symbols {
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
        symbol: &DocumentSymbol,
        container_name: Option<String>,
        url: &Url,
        res: &mut Vec<SymbolInformation>,
    ) {
        res.push(SymbolInformation {
            name: symbol.name.clone(),
            kind: symbol.kind,
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
    let _p = profile("handle_workspace_symbol");
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
            let container_name = nav.container_name().map(|v| v.to_string());
            let info = SymbolInformation {
                name: nav.name().to_string(),
                kind: to_proto::symbol_kind(nav.kind()),
                location: to_proto::location_from_nav(snap, nav)?,
                container_name,
                deprecated: None,
            };
            res.push(info);
        }
        Ok(res)
    }
}

pub(crate) fn handle_goto_definition(
    snap: GlobalStateSnapshot,
    params: lsp_types::GotoDefinitionParams,
) -> Result<Option<lsp_types::GotoDefinitionResponse>> {
    let _p = profile("handle_goto_definition");
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
    let _p = profile("handle_goto_implementation");
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
    let _p = profile("handle_goto_type_definition");
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
    let _p = profile("handle_parent_module");
    let position = from_proto::file_position(&snap, params)?;
    let navs = snap.analysis.parent_module(position)?;
    let res = to_proto::goto_definition_response(&snap, None, navs)?;
    Ok(Some(res))
}

pub(crate) fn handle_runnables(
    snap: GlobalStateSnapshot,
    params: lsp_ext::RunnablesParams,
) -> Result<Vec<lsp_ext::Runnable>> {
    let _p = profile("handle_runnables");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.analysis.file_line_index(file_id)?;
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
            if !runnable.nav.full_range().contains_inclusive(offset) {
                continue;
            }
        }
        if should_skip_target(&runnable, cargo_spec.as_ref()) {
            continue;
        }
        let mut runnable = to_proto::runnable(&snap, file_id, runnable)?;
        if expect_test {
            runnable.label = format!("{} + expect", runnable.label);
            runnable.args.expect_test = Some(true);
        }
        res.push(runnable);
    }

    // Add `cargo check` and `cargo test` for the whole package
    match cargo_spec {
        Some(spec) => {
            for &cmd in ["check", "test"].iter() {
                res.push(lsp_ext::Runnable {
                    label: format!("cargo {} -p {}", cmd, spec.package),
                    location: None,
                    kind: lsp_ext::RunnableKind::Cargo,
                    args: lsp_ext::CargoRunnable {
                        workspace_root: Some(spec.workspace_root.clone().into()),
                        cargo_args: vec![
                            cmd.to_string(),
                            "--package".to_string(),
                            spec.package.clone(),
                        ],
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
                    cargo_args: vec!["check".to_string(), "--workspace".to_string()],
                    executable_args: Vec::new(),
                    expect_test: None,
                },
            });
        }
    }
    Ok(res)
}

pub(crate) fn handle_completion(
    snap: GlobalStateSnapshot,
    params: lsp_types::CompletionParams,
) -> Result<Option<lsp_types::CompletionResponse>> {
    let _p = profile("handle_completion");
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

    let items = match snap.analysis.completions(&snap.config.completion, position)? {
        None => return Ok(None),
        Some(items) => items,
    };
    let line_index = snap.analysis.file_line_index(position.file_id)?;
    let line_endings = snap.file_line_endings(position.file_id);
    let items: Vec<CompletionItem> = items
        .into_iter()
        .map(|item| to_proto::completion_item(&line_index, line_endings, item))
        .collect();

    Ok(Some(items.into()))
}

pub(crate) fn handle_folding_range(
    snap: GlobalStateSnapshot,
    params: FoldingRangeParams,
) -> Result<Option<Vec<FoldingRange>>> {
    let _p = profile("handle_folding_range");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let folds = snap.analysis.folding_ranges(file_id)?;
    let text = snap.analysis.file_text(file_id)?;
    let line_index = snap.analysis.file_line_index(file_id)?;
    let line_folding_only = snap.config.client_caps.line_folding_only;
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
    let _p = profile("handle_signature_help");
    let position = from_proto::file_position(&snap, params.text_document_position_params)?;
    let call_info = match snap.analysis.call_info(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let concise = !snap.config.call_info_full;
    let mut active_parameter = call_info.active_parameter.map(|it| it as i64);
    if concise && call_info.signature.has_self_param {
        active_parameter = active_parameter.map(|it| it.saturating_sub(1));
    }
    let sig_info = to_proto::signature_information(call_info.signature, concise);

    Ok(Some(lsp_types::SignatureHelp {
        signatures: vec![sig_info],
        active_signature: Some(0),
        active_parameter,
    }))
}

pub(crate) fn handle_hover(
    snap: GlobalStateSnapshot,
    params: lsp_types::HoverParams,
) -> Result<Option<lsp_ext::Hover>> {
    let _p = profile("handle_hover");
    let position = from_proto::file_position(&snap, params.text_document_position_params)?;
    let info = match snap.analysis.hover(position)? {
        None => return Ok(None),
        Some(info) => info,
    };
    let line_index = snap.analysis.file_line_index(position.file_id)?;
    let range = to_proto::range(&line_index, info.range);
    let hover = lsp_ext::Hover {
        hover: lsp_types::Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: crate::markdown::format_docs(&info.info.to_markup()),
            }),
            range: Some(range),
        },
        actions: prepare_hover_actions(&snap, position.file_id, info.info.actions()),
    };

    Ok(Some(hover))
}

pub(crate) fn handle_prepare_rename(
    snap: GlobalStateSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> Result<Option<PrepareRenameResponse>> {
    let _p = profile("handle_prepare_rename");
    let position = from_proto::file_position(&snap, params)?;

    let optional_change = snap.analysis.rename(position, "dummy")?;
    let range = match optional_change {
        None => return Ok(None),
        Some(it) => it.range,
    };

    let line_index = snap.analysis.file_line_index(position.file_id)?;
    let range = to_proto::range(&line_index, range);
    Ok(Some(PrepareRenameResponse::Range(range)))
}

pub(crate) fn handle_rename(
    snap: GlobalStateSnapshot,
    params: RenameParams,
) -> Result<Option<WorkspaceEdit>> {
    let _p = profile("handle_rename");
    let position = from_proto::file_position(&snap, params.text_document_position)?;

    if params.new_name.is_empty() {
        return Err(LspError::new(
            ErrorCode::InvalidParams as i32,
            "New Name cannot be empty".into(),
        )
        .into());
    }

    let optional_change = snap.analysis.rename(position, &*params.new_name)?;
    let source_change = match optional_change {
        None => return Ok(None),
        Some(it) => it.info,
    };
    let workspace_edit = to_proto::workspace_edit(&snap, source_change)?;
    Ok(Some(workspace_edit))
}

pub(crate) fn handle_references(
    snap: GlobalStateSnapshot,
    params: lsp_types::ReferenceParams,
) -> Result<Option<Vec<Location>>> {
    let _p = profile("handle_references");
    let position = from_proto::file_position(&snap, params.text_document_position)?;

    let refs = match snap.analysis.find_all_refs(position, None)? {
        None => return Ok(None),
        Some(refs) => refs,
    };

    let locations = if params.context.include_declaration {
        refs.into_iter()
            .filter_map(|reference| to_proto::location(&snap, reference.file_range).ok())
            .collect()
    } else {
        // Only iterate over the references if include_declaration was false
        refs.references()
            .iter()
            .filter_map(|reference| to_proto::location(&snap, reference.file_range).ok())
            .collect()
    };

    Ok(Some(locations))
}

pub(crate) fn handle_formatting(
    snap: GlobalStateSnapshot,
    params: DocumentFormattingParams,
) -> Result<Option<Vec<lsp_types::TextEdit>>> {
    let _p = profile("handle_formatting");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let file = snap.analysis.file_text(file_id)?;
    let crate_ids = snap.analysis.crate_for(file_id)?;

    let file_line_index = snap.analysis.file_line_index(file_id)?;
    let end_position = to_proto::position(&file_line_index, TextSize::of(file.as_str()));

    let mut rustfmt = match &snap.config.rustfmt {
        RustfmtConfig::Rustfmt { extra_args } => {
            let mut cmd = process::Command::new(ra_toolchain::rustfmt());
            cmd.args(extra_args);
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

    if let Ok(path) = params.text_document.uri.to_file_path() {
        if let Some(parent) = path.parent() {
            rustfmt.current_dir(parent);
        }
    }
    let mut rustfmt = rustfmt.stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;

    rustfmt.stdin.as_mut().unwrap().write_all(file.as_bytes())?;

    let output = rustfmt.wait_with_output()?;
    let captured_stdout = String::from_utf8(output.stdout)?;

    if !output.status.success() {
        match output.status.code() {
            Some(1) => {
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
                           stdout: {}"#,
                        output.status, captured_stdout,
                    ),
                )
                .into());
            }
        }
    }

    Ok(Some(vec![lsp_types::TextEdit {
        range: Range::new(Position::new(0, 0), end_position),
        new_text: captured_stdout,
    }]))
}

fn handle_fixes(
    snap: &GlobalStateSnapshot,
    params: &lsp_types::CodeActionParams,
    res: &mut Vec<lsp_ext::CodeAction>,
) -> Result<()> {
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.analysis.file_line_index(file_id)?;
    let range = from_proto::text_range(&line_index, params.range);
    let diagnostics = snap.analysis.diagnostics(file_id)?;

    let fixes_from_diagnostics = diagnostics
        .into_iter()
        .filter_map(|d| Some((d.range, d.fix?)))
        .filter(|(diag_range, _fix)| diag_range.intersect(range).is_some())
        .map(|(_range, fix)| fix);
    for fix in fixes_from_diagnostics {
        let title = fix.label;
        let edit = to_proto::snippet_workspace_edit(&snap, fix.source_change)?;
        let action = lsp_ext::CodeAction {
            title,
            id: None,
            group: None,
            kind: Some(lsp_types::code_action_kind::QUICKFIX.into()),
            edit: Some(edit),
            command: None,
        };
        res.push(action);
    }

    for fix in snap.check_fixes.get(&file_id).into_iter().flatten() {
        let fix_range = from_proto::text_range(&line_index, fix.range);
        if fix_range.intersect(range).is_none() {
            continue;
        }
        res.push(fix.action.clone());
    }
    Ok(())
}

pub(crate) fn handle_code_action(
    snap: GlobalStateSnapshot,
    params: lsp_types::CodeActionParams,
) -> Result<Option<Vec<lsp_ext::CodeAction>>> {
    let _p = profile("handle_code_action");
    // We intentionally don't support command-based actions, as those either
    // requires custom client-code anyway, or requires server-initiated edits.
    // Server initiated edits break causality, so we avoid those as well.
    if !snap.config.client_caps.code_action_literals {
        return Ok(None);
    }

    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.analysis.file_line_index(file_id)?;
    let range = from_proto::text_range(&line_index, params.range);
    let frange = FileRange { file_id, range };
    let mut res: Vec<lsp_ext::CodeAction> = Vec::new();

    handle_fixes(&snap, &params, &mut res)?;

    if snap.config.client_caps.resolve_code_action {
        for (index, assist) in
            snap.analysis.unresolved_assists(&snap.config.assist, frange)?.into_iter().enumerate()
        {
            res.push(to_proto::unresolved_code_action(&snap, assist, index)?);
        }
    } else {
        for assist in snap.analysis.resolved_assists(&snap.config.assist, frange)?.into_iter() {
            res.push(to_proto::resolved_code_action(&snap, assist)?);
        }
    }

    Ok(Some(res))
}

pub(crate) fn handle_resolve_code_action(
    snap: GlobalStateSnapshot,
    params: lsp_ext::ResolveCodeActionParams,
) -> Result<Option<lsp_ext::SnippetWorkspaceEdit>> {
    let _p = profile("handle_resolve_code_action");
    let file_id = from_proto::file_id(&snap, &params.code_action_params.text_document.uri)?;
    let line_index = snap.analysis.file_line_index(file_id)?;
    let range = from_proto::text_range(&line_index, params.code_action_params.range);
    let frange = FileRange { file_id, range };

    let assists = snap.analysis.resolved_assists(&snap.config.assist, frange)?;
    let (id_string, index) = split_delim(&params.id, ':').unwrap();
    let index = index.parse::<usize>().unwrap();
    let assist = &assists[index];
    assert!(assist.assist.id.0 == id_string);
    Ok(to_proto::resolved_code_action(&snap, assist.clone())?.edit)
}

pub(crate) fn handle_code_lens(
    snap: GlobalStateSnapshot,
    params: lsp_types::CodeLensParams,
) -> Result<Option<Vec<CodeLens>>> {
    let _p = profile("handle_code_lens");
    let mut lenses: Vec<CodeLens> = Default::default();

    if snap.config.lens.none() {
        // early return before any db query!
        return Ok(Some(lenses));
    }

    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.analysis.file_line_index(file_id)?;
    let cargo_spec = CargoTargetSpec::for_file(&snap, file_id)?;

    if snap.config.lens.runnable() {
        // Gather runnables
        for runnable in snap.analysis.runnables(file_id)? {
            if should_skip_target(&runnable, cargo_spec.as_ref()) {
                continue;
            }

            let action = runnable.action();
            let range = to_proto::range(&line_index, runnable.nav.range());
            let r = to_proto::runnable(&snap, file_id, runnable)?;
            if snap.config.lens.run {
                let lens = CodeLens {
                    range,
                    command: Some(run_single_command(&r, action.run_title)),
                    data: None,
                };
                lenses.push(lens);
            }

            if action.debugee && snap.config.lens.debug {
                let debug_lens =
                    CodeLens { range, command: Some(debug_single_command(&r)), data: None };
                lenses.push(debug_lens);
            }
        }
    }

    if snap.config.lens.implementations {
        // Handle impls
        lenses.extend(
            snap.analysis
                .file_structure(file_id)?
                .into_iter()
                .filter(|it| match it.kind {
                    SyntaxKind::TRAIT_DEF | SyntaxKind::STRUCT_DEF | SyntaxKind::ENUM_DEF => true,
                    _ => false,
                })
                .map(|it| {
                    let range = to_proto::range(&line_index, it.node_range);
                    let pos = range.start;
                    let lens_params = lsp_types::request::GotoImplementationParams {
                        text_document_position_params: lsp_types::TextDocumentPositionParams::new(
                            params.text_document.clone(),
                            pos,
                        ),
                        work_done_progress_params: Default::default(),
                        partial_result_params: Default::default(),
                    };
                    CodeLens {
                        range,
                        command: None,
                        data: Some(to_value(CodeLensResolveData::Impls(lens_params)).unwrap()),
                    }
                }),
        );
    }
    Ok(Some(lenses))
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
enum CodeLensResolveData {
    Impls(lsp_types::request::GotoImplementationParams),
}

pub(crate) fn handle_code_lens_resolve(
    snap: GlobalStateSnapshot,
    code_lens: CodeLens,
) -> Result<CodeLens> {
    let _p = profile("handle_code_lens_resolve");
    let data = code_lens.data.unwrap();
    let resolve = from_json::<Option<CodeLensResolveData>>("CodeLensResolveData", data)?;
    match resolve {
        Some(CodeLensResolveData::Impls(lens_params)) => {
            let locations: Vec<Location> =
                match handle_goto_implementation(snap, lens_params.clone())? {
                    Some(lsp_types::GotoDefinitionResponse::Scalar(loc)) => vec![loc],
                    Some(lsp_types::GotoDefinitionResponse::Array(locs)) => locs,
                    Some(lsp_types::GotoDefinitionResponse::Link(links)) => links
                        .into_iter()
                        .map(|link| Location::new(link.target_uri, link.target_selection_range))
                        .collect(),
                    _ => vec![],
                };

            let title = implementation_title(locations.len());
            let cmd = show_references_command(
                title,
                &lens_params.text_document_position_params.text_document.uri,
                code_lens.range.start,
                locations,
            );
            Ok(CodeLens { range: code_lens.range, command: Some(cmd), data: None })
        }
        None => Ok(CodeLens {
            range: code_lens.range,
            command: Some(Command { title: "Error".into(), ..Default::default() }),
            data: None,
        }),
    }
}

pub(crate) fn handle_document_highlight(
    snap: GlobalStateSnapshot,
    params: lsp_types::DocumentHighlightParams,
) -> Result<Option<Vec<DocumentHighlight>>> {
    let _p = profile("handle_document_highlight");
    let position = from_proto::file_position(&snap, params.text_document_position_params)?;
    let line_index = snap.analysis.file_line_index(position.file_id)?;

    let refs = match snap
        .analysis
        .find_all_refs(position, Some(SearchScope::single_file(position.file_id)))?
    {
        None => return Ok(None),
        Some(refs) => refs,
    };

    let res = refs
        .into_iter()
        .filter(|reference| reference.file_range.file_id == position.file_id)
        .map(|reference| DocumentHighlight {
            range: to_proto::range(&line_index, reference.file_range.range),
            kind: reference.access.map(to_proto::document_highlight_kind),
        })
        .collect();
    Ok(Some(res))
}

pub(crate) fn handle_ssr(
    snap: GlobalStateSnapshot,
    params: lsp_ext::SsrParams,
) -> Result<lsp_types::WorkspaceEdit> {
    let _p = profile("handle_ssr");
    let source_change =
        snap.analysis.structural_search_replace(&params.query, params.parse_only)??;
    to_proto::workspace_edit(&snap, source_change)
}

pub(crate) fn publish_diagnostics(
    snap: &GlobalStateSnapshot,
    file_id: FileId,
) -> Result<Vec<Diagnostic>> {
    let _p = profile("publish_diagnostics");
    let line_index = snap.analysis.file_line_index(file_id)?;
    let diagnostics: Vec<Diagnostic> = snap
        .analysis
        .diagnostics(file_id)?
        .into_iter()
        .map(|d| Diagnostic {
            range: to_proto::range(&line_index, d.range),
            severity: Some(to_proto::diagnostic_severity(d.severity)),
            code: None,
            source: Some("rust-analyzer".to_string()),
            message: d.message,
            related_information: None,
            tags: None,
        })
        .collect();
    Ok(diagnostics)
}

pub(crate) fn handle_inlay_hints(
    snap: GlobalStateSnapshot,
    params: InlayHintsParams,
) -> Result<Vec<InlayHint>> {
    let _p = profile("handle_inlay_hints");
    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let line_index = snap.analysis.file_line_index(file_id)?;
    Ok(snap
        .analysis
        .inlay_hints(file_id, &snap.config.inlay_hints)?
        .into_iter()
        .map(|it| to_proto::inlay_int(&line_index, it))
        .collect())
}

pub(crate) fn handle_call_hierarchy_prepare(
    snap: GlobalStateSnapshot,
    params: CallHierarchyPrepareParams,
) -> Result<Option<Vec<CallHierarchyItem>>> {
    let _p = profile("handle_call_hierarchy_prepare");
    let position = from_proto::file_position(&snap, params.text_document_position_params)?;

    let nav_info = match snap.analysis.call_hierarchy(position)? {
        None => return Ok(None),
        Some(it) => it,
    };

    let RangeInfo { range: _, info: navs } = nav_info;
    let res = navs
        .into_iter()
        .filter(|it| it.kind() == SyntaxKind::FN_DEF)
        .map(|it| to_proto::call_hierarchy_item(&snap, it))
        .collect::<Result<Vec<_>>>()?;

    Ok(Some(res))
}

pub(crate) fn handle_call_hierarchy_incoming(
    snap: GlobalStateSnapshot,
    params: CallHierarchyIncomingCallsParams,
) -> Result<Option<Vec<CallHierarchyIncomingCall>>> {
    let _p = profile("handle_call_hierarchy_incoming");
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
        let file_id = call_item.target.file_id();
        let line_index = snap.analysis.file_line_index(file_id)?;
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
    let _p = profile("handle_call_hierarchy_outgoing");
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
        let file_id = call_item.target.file_id();
        let line_index = snap.analysis.file_line_index(file_id)?;
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

pub(crate) fn handle_semantic_tokens(
    snap: GlobalStateSnapshot,
    params: SemanticTokensParams,
) -> Result<Option<SemanticTokensResult>> {
    let _p = profile("handle_semantic_tokens");

    let file_id = from_proto::file_id(&snap, &params.text_document.uri)?;
    let text = snap.analysis.file_text(file_id)?;
    let line_index = snap.analysis.file_line_index(file_id)?;

    let highlights = snap.analysis.highlight(file_id)?;
    let semantic_tokens = to_proto::semantic_tokens(&text, &line_index, highlights);
    Ok(Some(semantic_tokens.into()))
}

pub(crate) fn handle_semantic_tokens_range(
    snap: GlobalStateSnapshot,
    params: SemanticTokensRangeParams,
) -> Result<Option<SemanticTokensRangeResult>> {
    let _p = profile("handle_semantic_tokens_range");

    let frange = from_proto::file_range(&snap, params.text_document, params.range)?;
    let text = snap.analysis.file_text(frange.file_id)?;
    let line_index = snap.analysis.file_line_index(frange.file_id)?;

    let highlights = snap.analysis.highlight_range(frange)?;
    let semantic_tokens = to_proto::semantic_tokens(&text, &line_index, highlights);
    Ok(Some(semantic_tokens.into()))
}

fn implementation_title(count: usize) -> String {
    if count == 1 {
        "1 implementation".into()
    } else {
        format!("{} implementations", count)
    }
}

fn show_references_command(
    title: String,
    uri: &lsp_types::Url,
    position: lsp_types::Position,
    locations: Vec<lsp_types::Location>,
) -> Command {
    // We cannot use the 'editor.action.showReferences' command directly
    // because that command requires vscode types which we convert in the handler
    // on the client side.

    Command {
        title,
        command: "rust-analyzer.showReferences".into(),
        arguments: Some(vec![
            to_value(uri).unwrap(),
            to_value(position).unwrap(),
            to_value(locations).unwrap(),
        ]),
    }
}

fn run_single_command(runnable: &lsp_ext::Runnable, title: &str) -> Command {
    Command {
        title: title.to_string(),
        command: "rust-analyzer.runSingle".into(),
        arguments: Some(vec![to_value(runnable).unwrap()]),
    }
}

fn debug_single_command(runnable: &lsp_ext::Runnable) -> Command {
    Command {
        title: "Debug".into(),
        command: "rust-analyzer.debugSingle".into(),
        arguments: Some(vec![to_value(runnable).unwrap()]),
    }
}

fn goto_location_command(snap: &GlobalStateSnapshot, nav: &NavigationTarget) -> Option<Command> {
    let value = if snap.config.client_caps.location_link {
        let link = to_proto::location_link(snap, None, nav.clone()).ok()?;
        to_value(link).ok()?
    } else {
        let range = FileRange { file_id: nav.file_id(), range: nav.range() };
        let location = to_proto::location(snap, range).ok()?;
        to_value(location).ok()?
    };

    Some(Command {
        title: nav.name().to_string(),
        command: "rust-analyzer.gotoLocation".into(),
        arguments: Some(vec![value]),
    })
}

fn to_command_link(command: Command, tooltip: String) -> lsp_ext::CommandLink {
    lsp_ext::CommandLink { tooltip: Some(tooltip), command }
}

fn show_impl_command_link(
    snap: &GlobalStateSnapshot,
    position: &FilePosition,
) -> Option<lsp_ext::CommandLinkGroup> {
    if snap.config.hover.implementations {
        if let Some(nav_data) = snap.analysis.goto_implementation(*position).unwrap_or(None) {
            let uri = to_proto::url(snap, position.file_id);
            let line_index = snap.analysis.file_line_index(position.file_id).ok()?;
            let position = to_proto::position(&line_index, position.offset);
            let locations: Vec<_> = nav_data
                .info
                .into_iter()
                .filter_map(|nav| to_proto::location_from_nav(snap, nav).ok())
                .collect();
            let title = implementation_title(locations.len());
            let command = show_references_command(title, &uri, position, locations);

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
    file_id: FileId,
    runnable: Runnable,
) -> Option<lsp_ext::CommandLinkGroup> {
    let cargo_spec = CargoTargetSpec::for_file(&snap, file_id).ok()?;
    if !snap.config.hover.runnable() || should_skip_target(&runnable, cargo_spec.as_ref()) {
        return None;
    }

    let action: &'static _ = runnable.action();
    to_proto::runnable(snap, file_id, runnable).ok().map(|r| {
        let mut group = lsp_ext::CommandLinkGroup::default();

        if snap.config.hover.run {
            let run_command = run_single_command(&r, action.run_title);
            group.commands.push(to_command_link(run_command, r.label.clone()));
        }

        if snap.config.hover.debug {
            let dbg_command = debug_single_command(&r);
            group.commands.push(to_command_link(dbg_command, r.label));
        }

        group
    })
}

fn goto_type_action_links(
    snap: &GlobalStateSnapshot,
    nav_targets: &[HoverGotoTypeData],
) -> Option<lsp_ext::CommandLinkGroup> {
    if !snap.config.hover.goto_type_def || nav_targets.is_empty() {
        return None;
    }

    Some(lsp_ext::CommandLinkGroup {
        title: Some("Go to ".into()),
        commands: nav_targets
            .iter()
            .filter_map(|it| {
                goto_location_command(snap, &it.nav)
                    .map(|cmd| to_command_link(cmd, it.mod_path.clone()))
            })
            .collect(),
    })
}

fn prepare_hover_actions(
    snap: &GlobalStateSnapshot,
    file_id: FileId,
    actions: &[HoverAction],
) -> Vec<lsp_ext::CommandLinkGroup> {
    if snap.config.hover.none() || !snap.config.client_caps.hover_actions {
        return Vec::new();
    }

    actions
        .iter()
        .filter_map(|it| match it {
            HoverAction::Implementaion(position) => show_impl_command_link(snap, position),
            HoverAction::Runnable(r) => runnable_action_links(snap, file_id, r.clone()),
            HoverAction::GoToType(targets) => goto_type_action_links(snap, targets),
        })
        .collect()
}

fn should_skip_target(runnable: &Runnable, cargo_spec: Option<&CargoTargetSpec>) -> bool {
    match runnable.kind {
        RunnableKind::Bin => {
            // Do not suggest binary run on other target than binary
            match &cargo_spec {
                Some(spec) => !matches!(spec.target_kind, TargetKind::Bin | TargetKind::Example),
                None => true,
            }
        }
        _ => false,
    }
}
