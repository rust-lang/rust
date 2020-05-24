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
    DocumentSymbol, FoldingRange, FoldingRangeParams, Hover, HoverContents, Location,
    MarkupContent, MarkupKind, Position, PrepareRenameResponse, Range, RenameParams,
    SemanticTokensParams, SemanticTokensRangeParams, SemanticTokensRangeResult,
    SemanticTokensResult, SymbolInformation, TextDocumentIdentifier, Url, WorkspaceEdit,
};
use ra_cfg::CfgExpr;
use ra_ide::{
    FileId, FilePosition, FileRange, Query, RangeInfo, Runnable, RunnableKind, SearchScope,
    TextEdit,
};
use ra_prof::profile;
use ra_project_model::TargetKind;
use ra_syntax::{AstNode, SmolStr, SyntaxKind, TextRange, TextSize};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::to_value;
use stdx::format_to;

use crate::{
    cargo_target_spec::CargoTargetSpec,
    config::RustfmtConfig,
    diagnostics::DiagnosticTask,
    from_json, from_proto,
    lsp_ext::{self, InlayHint, InlayHintsParams},
    to_proto,
    world::WorldSnapshot,
    LspError, Result,
};

pub fn handle_analyzer_status(world: WorldSnapshot, _: ()) -> Result<String> {
    let _p = profile("handle_analyzer_status");
    let mut buf = world.status();
    format_to!(buf, "\n\nrequests:\n");
    let requests = world.latest_requests.read();
    for (is_last, r) in requests.iter() {
        let mark = if is_last { "*" } else { " " };
        format_to!(buf, "{}{:4} {:<36}{}ms\n", mark, r.id, r.method, r.duration.as_millis());
    }
    Ok(buf)
}

pub fn handle_syntax_tree(
    world: WorldSnapshot,
    params: lsp_ext::SyntaxTreeParams,
) -> Result<String> {
    let _p = profile("handle_syntax_tree");
    let id = from_proto::file_id(&world, &params.text_document.uri)?;
    let line_index = world.analysis().file_line_index(id)?;
    let text_range = params.range.map(|r| from_proto::text_range(&line_index, r));
    let res = world.analysis().syntax_tree(id, text_range)?;
    Ok(res)
}

pub fn handle_expand_macro(
    world: WorldSnapshot,
    params: lsp_ext::ExpandMacroParams,
) -> Result<Option<lsp_ext::ExpandedMacro>> {
    let _p = profile("handle_expand_macro");
    let file_id = from_proto::file_id(&world, &params.text_document.uri)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let offset = params.position.map(|p| from_proto::offset(&line_index, p));

    match offset {
        None => Ok(None),
        Some(offset) => {
            let res = world.analysis().expand_macro(FilePosition { file_id, offset })?;
            Ok(res.map(|it| lsp_ext::ExpandedMacro { name: it.name, expansion: it.expansion }))
        }
    }
}

pub fn handle_selection_range(
    world: WorldSnapshot,
    params: lsp_types::SelectionRangeParams,
) -> Result<Option<Vec<lsp_types::SelectionRange>>> {
    let _p = profile("handle_selection_range");
    let file_id = from_proto::file_id(&world, &params.text_document.uri)?;
    let line_index = world.analysis().file_line_index(file_id)?;
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
                    let next = world.analysis().extend_selection(frange)?;
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

pub fn handle_find_matching_brace(
    world: WorldSnapshot,
    params: lsp_ext::FindMatchingBraceParams,
) -> Result<Vec<Position>> {
    let _p = profile("handle_find_matching_brace");
    let file_id = from_proto::file_id(&world, &params.text_document.uri)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let res = params
        .offsets
        .into_iter()
        .map(|position| {
            let offset = from_proto::offset(&line_index, position);
            let offset = match world.analysis().matching_brace(FilePosition { file_id, offset }) {
                Ok(Some(matching_brace_offset)) => matching_brace_offset,
                Err(_) | Ok(None) => offset,
            };
            to_proto::position(&line_index, offset)
        })
        .collect();
    Ok(res)
}

pub fn handle_join_lines(
    world: WorldSnapshot,
    params: lsp_ext::JoinLinesParams,
) -> Result<Vec<lsp_types::TextEdit>> {
    let _p = profile("handle_join_lines");
    let file_id = from_proto::file_id(&world, &params.text_document.uri)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let line_endings = world.file_line_endings(file_id);
    let mut res = TextEdit::default();
    for range in params.ranges {
        let range = from_proto::text_range(&line_index, range);
        let edit = world.analysis().join_lines(FileRange { file_id, range })?;
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

pub fn handle_on_enter(
    world: WorldSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> Result<Option<lsp_ext::SnippetWorkspaceEdit>> {
    let _p = profile("handle_on_enter");
    let position = from_proto::file_position(&world, params)?;
    match world.analysis().on_enter(position)? {
        None => Ok(None),
        Some(source_change) => to_proto::snippet_workspace_edit(&world, source_change).map(Some),
    }
}

// Don't forget to add new trigger characters to `ServerCapabilities` in `caps.rs`.
pub fn handle_on_type_formatting(
    world: WorldSnapshot,
    params: lsp_types::DocumentOnTypeFormattingParams,
) -> Result<Option<Vec<lsp_types::TextEdit>>> {
    let _p = profile("handle_on_type_formatting");
    let mut position = from_proto::file_position(&world, params.text_document_position)?;
    let line_index = world.analysis().file_line_index(position.file_id)?;
    let line_endings = world.file_line_endings(position.file_id);

    // in `ra_ide`, the `on_type` invariant is that
    // `text.char_at(position) == typed_char`.
    position.offset -= TextSize::of('.');
    let char_typed = params.ch.chars().next().unwrap_or('\0');
    assert!({
        let text = world.analysis().file_text(position.file_id)?;
        text[usize::from(position.offset)..].starts_with(char_typed)
    });

    // We have an assist that inserts ` ` after typing `->` in `fn foo() ->{`,
    // but it requires precise cursor positioning to work, and one can't
    // position the cursor with on_type formatting. So, let's just toggle this
    // feature off here, hoping that we'll enable it one day, 😿.
    if char_typed == '>' {
        return Ok(None);
    }

    let edit = world.analysis().on_char_typed(position, char_typed)?;
    let mut edit = match edit {
        Some(it) => it,
        None => return Ok(None),
    };

    // This should be a single-file edit
    let edit = edit.source_file_edits.pop().unwrap();

    let change = to_proto::text_edit_vec(&line_index, line_endings, edit.edit);
    Ok(Some(change))
}

pub fn handle_document_symbol(
    world: WorldSnapshot,
    params: lsp_types::DocumentSymbolParams,
) -> Result<Option<lsp_types::DocumentSymbolResponse>> {
    let _p = profile("handle_document_symbol");
    let file_id = from_proto::file_id(&world, &params.text_document.uri)?;
    let line_index = world.analysis().file_line_index(file_id)?;

    let mut parents: Vec<(DocumentSymbol, Option<usize>)> = Vec::new();

    for symbol in world.analysis().file_structure(file_id)? {
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

    let res = if world.config.client_caps.hierarchical_symbols {
        document_symbols.into()
    } else {
        let url = to_proto::url(&world, file_id)?;
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

pub fn handle_workspace_symbol(
    world: WorldSnapshot,
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
    let mut res = exec_query(&world, query)?;
    if res.is_empty() && !all_symbols {
        let mut query = Query::new(params.query);
        query.limit(128);
        res = exec_query(&world, query)?;
    }

    return Ok(Some(res));

    fn exec_query(world: &WorldSnapshot, query: Query) -> Result<Vec<SymbolInformation>> {
        let mut res = Vec::new();
        for nav in world.analysis().symbol_search(query)? {
            let info = SymbolInformation {
                name: nav.name().to_string(),
                kind: to_proto::symbol_kind(nav.kind()),
                location: to_proto::location(world, nav.file_range())?,
                container_name: nav.container_name().map(|v| v.to_string()),
                deprecated: None,
            };
            res.push(info);
        }
        Ok(res)
    }
}

pub fn handle_goto_definition(
    world: WorldSnapshot,
    params: lsp_types::GotoDefinitionParams,
) -> Result<Option<lsp_types::GotoDefinitionResponse>> {
    let _p = profile("handle_goto_definition");
    let position = from_proto::file_position(&world, params.text_document_position_params)?;
    let nav_info = match world.analysis().goto_definition(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let res = to_proto::goto_definition_response(
        &world,
        FileRange { file_id: position.file_id, range: nav_info.range },
        nav_info.info,
    )?;
    Ok(Some(res))
}

pub fn handle_goto_implementation(
    world: WorldSnapshot,
    params: lsp_types::request::GotoImplementationParams,
) -> Result<Option<lsp_types::request::GotoImplementationResponse>> {
    let _p = profile("handle_goto_implementation");
    let position = from_proto::file_position(&world, params.text_document_position_params)?;
    let nav_info = match world.analysis().goto_implementation(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let res = to_proto::goto_definition_response(
        &world,
        FileRange { file_id: position.file_id, range: nav_info.range },
        nav_info.info,
    )?;
    Ok(Some(res))
}

pub fn handle_goto_type_definition(
    world: WorldSnapshot,
    params: lsp_types::request::GotoTypeDefinitionParams,
) -> Result<Option<lsp_types::request::GotoTypeDefinitionResponse>> {
    let _p = profile("handle_goto_type_definition");
    let position = from_proto::file_position(&world, params.text_document_position_params)?;
    let nav_info = match world.analysis().goto_type_definition(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let res = to_proto::goto_definition_response(
        &world,
        FileRange { file_id: position.file_id, range: nav_info.range },
        nav_info.info,
    )?;
    Ok(Some(res))
}

pub fn handle_parent_module(
    world: WorldSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> Result<Vec<Location>> {
    let _p = profile("handle_parent_module");
    let position = from_proto::file_position(&world, params)?;
    world
        .analysis()
        .parent_module(position)?
        .into_iter()
        .map(|it| to_proto::location(&world, it.file_range()))
        .collect::<Result<Vec<_>>>()
}

pub fn handle_runnables(
    world: WorldSnapshot,
    params: lsp_ext::RunnablesParams,
) -> Result<Vec<lsp_ext::Runnable>> {
    let _p = profile("handle_runnables");
    let file_id = from_proto::file_id(&world, &params.text_document.uri)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let offset = params.position.map(|it| from_proto::offset(&line_index, it));
    let mut res = Vec::new();
    let workspace_root = world.workspace_root_for(file_id);
    let cargo_spec = CargoTargetSpec::for_file(&world, file_id)?;
    for runnable in world.analysis().runnables(file_id)? {
        if let Some(offset) = offset {
            if !runnable.range.contains_inclusive(offset) {
                continue;
            }
        }
        // Do not suggest binary run on other target than binary
        if let RunnableKind::Bin = runnable.kind {
            if let Some(spec) = &cargo_spec {
                match spec.target_kind {
                    TargetKind::Bin => {}
                    _ => continue,
                }
            }
        }
        res.push(to_lsp_runnable(&world, file_id, runnable)?);
    }

    // Add `cargo check` and `cargo test` for the whole package
    match cargo_spec {
        Some(spec) => {
            for &cmd in ["check", "test"].iter() {
                res.push(lsp_ext::Runnable {
                    range: Default::default(),
                    label: format!("cargo {} -p {}", cmd, spec.package),
                    bin: "cargo".to_string(),
                    args: vec![cmd.to_string(), "--package".to_string(), spec.package.clone()],
                    extra_args: Vec::new(),
                    env: FxHashMap::default(),
                    cwd: workspace_root.map(|root| root.to_owned()),
                })
            }
        }
        None => {
            res.push(lsp_ext::Runnable {
                range: Default::default(),
                label: "cargo check --workspace".to_string(),
                bin: "cargo".to_string(),
                args: vec!["check".to_string(), "--workspace".to_string()],
                extra_args: Vec::new(),
                env: FxHashMap::default(),
                cwd: workspace_root.map(|root| root.to_owned()),
            });
        }
    }
    Ok(res)
}

pub fn handle_completion(
    world: WorldSnapshot,
    params: lsp_types::CompletionParams,
) -> Result<Option<lsp_types::CompletionResponse>> {
    let _p = profile("handle_completion");
    let position = from_proto::file_position(&world, params.text_document_position)?;
    let completion_triggered_after_single_colon = {
        let mut res = false;
        if let Some(ctx) = params.context {
            if ctx.trigger_character.unwrap_or_default() == ":" {
                let source_file = world.analysis().parse(position.file_id)?;
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

    let items = match world.analysis().completions(&world.config.completion, position)? {
        None => return Ok(None),
        Some(items) => items,
    };
    let line_index = world.analysis().file_line_index(position.file_id)?;
    let line_endings = world.file_line_endings(position.file_id);
    let items: Vec<CompletionItem> = items
        .into_iter()
        .map(|item| to_proto::completion_item(&line_index, line_endings, item))
        .collect();

    Ok(Some(items.into()))
}

pub fn handle_folding_range(
    world: WorldSnapshot,
    params: FoldingRangeParams,
) -> Result<Option<Vec<FoldingRange>>> {
    let _p = profile("handle_folding_range");
    let file_id = from_proto::file_id(&world, &params.text_document.uri)?;
    let folds = world.analysis().folding_ranges(file_id)?;
    let text = world.analysis().file_text(file_id)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let line_folding_only = world.config.client_caps.line_folding_only;
    let res = folds
        .into_iter()
        .map(|it| to_proto::folding_range(&*text, &line_index, line_folding_only, it))
        .collect();
    Ok(Some(res))
}

pub fn handle_signature_help(
    world: WorldSnapshot,
    params: lsp_types::SignatureHelpParams,
) -> Result<Option<lsp_types::SignatureHelp>> {
    let _p = profile("handle_signature_help");
    let position = from_proto::file_position(&world, params.text_document_position_params)?;
    let call_info = match world.analysis().call_info(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let concise = !world.config.call_info_full;
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

pub fn handle_hover(world: WorldSnapshot, params: lsp_types::HoverParams) -> Result<Option<Hover>> {
    let _p = profile("handle_hover");
    let position = from_proto::file_position(&world, params.text_document_position_params)?;
    let info = match world.analysis().hover(position)? {
        None => return Ok(None),
        Some(info) => info,
    };
    let line_index = world.analysis.file_line_index(position.file_id)?;
    let range = to_proto::range(&line_index, info.range);
    let res = Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: crate::markdown::format_docs(&info.info.to_markup()),
        }),
        range: Some(range),
    };
    Ok(Some(res))
}

pub fn handle_prepare_rename(
    world: WorldSnapshot,
    params: lsp_types::TextDocumentPositionParams,
) -> Result<Option<PrepareRenameResponse>> {
    let _p = profile("handle_prepare_rename");
    let position = from_proto::file_position(&world, params)?;

    let optional_change = world.analysis().rename(position, "dummy")?;
    let range = match optional_change {
        None => return Ok(None),
        Some(it) => it.range,
    };

    let line_index = world.analysis().file_line_index(position.file_id)?;
    let range = to_proto::range(&line_index, range);
    Ok(Some(PrepareRenameResponse::Range(range)))
}

pub fn handle_rename(world: WorldSnapshot, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
    let _p = profile("handle_rename");
    let position = from_proto::file_position(&world, params.text_document_position)?;

    if params.new_name.is_empty() {
        return Err(LspError::new(
            ErrorCode::InvalidParams as i32,
            "New Name cannot be empty".into(),
        )
        .into());
    }

    let optional_change = world.analysis().rename(position, &*params.new_name)?;
    let source_change = match optional_change {
        None => return Ok(None),
        Some(it) => it.info,
    };
    let workspace_edit = to_proto::workspace_edit(&world, source_change)?;
    Ok(Some(workspace_edit))
}

pub fn handle_references(
    world: WorldSnapshot,
    params: lsp_types::ReferenceParams,
) -> Result<Option<Vec<Location>>> {
    let _p = profile("handle_references");
    let position = from_proto::file_position(&world, params.text_document_position)?;

    let refs = match world.analysis().find_all_refs(position, None)? {
        None => return Ok(None),
        Some(refs) => refs,
    };

    let locations = if params.context.include_declaration {
        refs.into_iter()
            .filter_map(|reference| to_proto::location(&world, reference.file_range).ok())
            .collect()
    } else {
        // Only iterate over the references if include_declaration was false
        refs.references()
            .iter()
            .filter_map(|reference| to_proto::location(&world, reference.file_range).ok())
            .collect()
    };

    Ok(Some(locations))
}

pub fn handle_formatting(
    world: WorldSnapshot,
    params: DocumentFormattingParams,
) -> Result<Option<Vec<lsp_types::TextEdit>>> {
    let _p = profile("handle_formatting");
    let file_id = from_proto::file_id(&world, &params.text_document.uri)?;
    let file = world.analysis().file_text(file_id)?;
    let crate_ids = world.analysis().crate_for(file_id)?;

    let file_line_index = world.analysis().file_line_index(file_id)?;
    let end_position = to_proto::position(&file_line_index, TextSize::of(file.as_str()));

    let mut rustfmt = match &world.config.rustfmt {
        RustfmtConfig::Rustfmt { extra_args } => {
            let mut cmd = process::Command::new("rustfmt");
            cmd.args(extra_args);
            if let Some(&crate_id) = crate_ids.first() {
                // Assume all crates are in the same edition
                let edition = world.analysis().crate_edition(crate_id)?;
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

pub fn handle_code_action(
    world: WorldSnapshot,
    params: lsp_types::CodeActionParams,
) -> Result<Option<Vec<lsp_ext::CodeAction>>> {
    let _p = profile("handle_code_action");
    // We intentionally don't support command-based actions, as those either
    // requires custom client-code anyway, or requires server-initiated edits.
    // Server initiated edits break causality, so we avoid those as well.
    if !world.config.client_caps.code_action_literals {
        return Ok(None);
    }

    let file_id = from_proto::file_id(&world, &params.text_document.uri)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let range = from_proto::text_range(&line_index, params.range);
    let frange = FileRange { file_id, range };

    let diagnostics = world.analysis().diagnostics(file_id)?;
    let mut res: Vec<lsp_ext::CodeAction> = Vec::new();

    let fixes_from_diagnostics = diagnostics
        .into_iter()
        .filter_map(|d| Some((d.range, d.fix?)))
        .filter(|(diag_range, _fix)| diag_range.intersect(range).is_some())
        .map(|(_range, fix)| fix);

    for fix in fixes_from_diagnostics {
        let title = fix.label;
        let edit = to_proto::snippet_workspace_edit(&world, fix.source_change)?;
        let action =
            lsp_ext::CodeAction { title, group: None, kind: None, edit: Some(edit), command: None };
        res.push(action);
    }

    for fix in world.check_fixes.get(&file_id).into_iter().flatten() {
        let fix_range = from_proto::text_range(&line_index, fix.range);
        if fix_range.intersect(range).is_none() {
            continue;
        }
        res.push(fix.action.clone());
    }

    for assist in world.analysis().assists(&world.config.assist, frange)?.into_iter() {
        res.push(to_proto::code_action(&world, assist)?.into());
    }
    Ok(Some(res))
}

pub fn handle_code_lens(
    world: WorldSnapshot,
    params: lsp_types::CodeLensParams,
) -> Result<Option<Vec<CodeLens>>> {
    let _p = profile("handle_code_lens");
    let mut lenses: Vec<CodeLens> = Default::default();

    if world.config.lens.none() {
        // early return before any db query!
        return Ok(Some(lenses));
    }

    let file_id = from_proto::file_id(&world, &params.text_document.uri)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let cargo_spec = CargoTargetSpec::for_file(&world, file_id)?;

    if world.config.lens.runnable() {
        // Gather runnables
        for runnable in world.analysis().runnables(file_id)? {
            let (run_title, debugee) = match &runnable.kind {
                RunnableKind::Test { .. } | RunnableKind::TestMod { .. } => {
                    ("▶\u{fe0e} Run Test", true)
                }
                RunnableKind::DocTest { .. } => {
                    // cargo does not support -no-run for doctests
                    ("▶\u{fe0e} Run Doctest", false)
                }
                RunnableKind::Bench { .. } => {
                    // Nothing wrong with bench debugging
                    ("Run Bench", true)
                }
                RunnableKind::Bin => {
                    // Do not suggest binary run on other target than binary
                    match &cargo_spec {
                        Some(spec) => match spec.target_kind {
                            TargetKind::Bin => ("Run", true),
                            _ => continue,
                        },
                        None => continue,
                    }
                }
            };

            let mut r = to_lsp_runnable(&world, file_id, runnable)?;
            if world.config.lens.run {
                let lens = CodeLens {
                    range: r.range,
                    command: Some(Command {
                        title: run_title.to_string(),
                        command: "rust-analyzer.runSingle".into(),
                        arguments: Some(vec![to_value(&r).unwrap()]),
                    }),
                    data: None,
                };
                lenses.push(lens);
            }

            if debugee && world.config.lens.debug {
                if r.args[0] == "run" {
                    r.args[0] = "build".into();
                } else {
                    r.args.push("--no-run".into());
                }
                let debug_lens = CodeLens {
                    range: r.range,
                    command: Some(Command {
                        title: "Debug".into(),
                        command: "rust-analyzer.debugSingle".into(),
                        arguments: Some(vec![to_value(r).unwrap()]),
                    }),
                    data: None,
                };
                lenses.push(debug_lens);
            }
        }
    }

    if world.config.lens.impementations {
        // Handle impls
        lenses.extend(
            world
                .analysis()
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

pub fn handle_code_lens_resolve(world: WorldSnapshot, code_lens: CodeLens) -> Result<CodeLens> {
    let _p = profile("handle_code_lens_resolve");
    let data = code_lens.data.unwrap();
    let resolve = from_json::<Option<CodeLensResolveData>>("CodeLensResolveData", data)?;
    match resolve {
        Some(CodeLensResolveData::Impls(lens_params)) => {
            let locations: Vec<Location> =
                match handle_goto_implementation(world, lens_params.clone())? {
                    Some(lsp_types::GotoDefinitionResponse::Scalar(loc)) => vec![loc],
                    Some(lsp_types::GotoDefinitionResponse::Array(locs)) => locs,
                    Some(lsp_types::GotoDefinitionResponse::Link(links)) => links
                        .into_iter()
                        .map(|link| Location::new(link.target_uri, link.target_selection_range))
                        .collect(),
                    _ => vec![],
                };

            let title = if locations.len() == 1 {
                "1 implementation".into()
            } else {
                format!("{} implementations", locations.len())
            };

            // We cannot use the 'editor.action.showReferences' command directly
            // because that command requires vscode types which we convert in the handler
            // on the client side.
            let cmd = Command {
                title,
                command: "rust-analyzer.showReferences".into(),
                arguments: Some(vec![
                    to_value(&lens_params.text_document_position_params.text_document.uri).unwrap(),
                    to_value(code_lens.range.start).unwrap(),
                    to_value(locations).unwrap(),
                ]),
            };
            Ok(CodeLens { range: code_lens.range, command: Some(cmd), data: None })
        }
        None => Ok(CodeLens {
            range: code_lens.range,
            command: Some(Command { title: "Error".into(), ..Default::default() }),
            data: None,
        }),
    }
}

pub fn handle_document_highlight(
    world: WorldSnapshot,
    params: lsp_types::DocumentHighlightParams,
) -> Result<Option<Vec<DocumentHighlight>>> {
    let _p = profile("handle_document_highlight");
    let position = from_proto::file_position(&world, params.text_document_position_params)?;
    let line_index = world.analysis().file_line_index(position.file_id)?;

    let refs = match world
        .analysis()
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

pub fn handle_ssr(
    world: WorldSnapshot,
    params: lsp_ext::SsrParams,
) -> Result<lsp_types::WorkspaceEdit> {
    let _p = profile("handle_ssr");
    let source_change =
        world.analysis().structural_search_replace(&params.query, params.parse_only)??;
    to_proto::workspace_edit(&world, source_change)
}

pub fn publish_diagnostics(world: &WorldSnapshot, file_id: FileId) -> Result<DiagnosticTask> {
    let _p = profile("publish_diagnostics");
    let line_index = world.analysis().file_line_index(file_id)?;
    let diagnostics: Vec<Diagnostic> = world
        .analysis()
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
    Ok(DiagnosticTask::SetNative(file_id, diagnostics))
}

fn to_lsp_runnable(
    world: &WorldSnapshot,
    file_id: FileId,
    runnable: Runnable,
) -> Result<lsp_ext::Runnable> {
    let spec = CargoTargetSpec::for_file(world, file_id)?;
    let target = spec.as_ref().map(|s| s.target.clone());
    let mut features_needed = vec![];
    for cfg_expr in &runnable.cfg_exprs {
        collect_minimal_features_needed(cfg_expr, &mut features_needed);
    }
    let (args, extra_args) =
        CargoTargetSpec::runnable_args(spec, &runnable.kind, &features_needed)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let label = match &runnable.kind {
        RunnableKind::Test { test_id, .. } => format!("test {}", test_id),
        RunnableKind::TestMod { path } => format!("test-mod {}", path),
        RunnableKind::Bench { test_id } => format!("bench {}", test_id),
        RunnableKind::DocTest { test_id, .. } => format!("doctest {}", test_id),
        RunnableKind::Bin => {
            target.map_or_else(|| "run binary".to_string(), |t| format!("run {}", t))
        }
    };
    Ok(lsp_ext::Runnable {
        range: to_proto::range(&line_index, runnable.range),
        label,
        bin: "cargo".to_string(),
        args,
        extra_args,
        env: {
            let mut m = FxHashMap::default();
            m.insert("RUST_BACKTRACE".to_string(), "short".to_string());
            m
        },
        cwd: world.workspace_root_for(file_id).map(|root| root.to_owned()),
    })
}

/// Fill minimal features needed
fn collect_minimal_features_needed(cfg_expr: &CfgExpr, features: &mut Vec<SmolStr>) {
    match cfg_expr {
        CfgExpr::KeyValue { key, value } if key == "feature" => features.push(value.clone()),
        CfgExpr::All(preds) => {
            preds.iter().for_each(|cfg| collect_minimal_features_needed(cfg, features));
        }
        CfgExpr::Any(preds) => {
            for cfg in preds {
                let len_features = features.len();
                collect_minimal_features_needed(cfg, features);
                if len_features != features.len() {
                    break;
                }
            }
        }
        _ => {}
    }
}

pub fn handle_inlay_hints(
    world: WorldSnapshot,
    params: InlayHintsParams,
) -> Result<Vec<InlayHint>> {
    let _p = profile("handle_inlay_hints");
    let file_id = from_proto::file_id(&world, &params.text_document.uri)?;
    let analysis = world.analysis();
    let line_index = analysis.file_line_index(file_id)?;
    Ok(analysis
        .inlay_hints(file_id, &world.config.inlay_hints)?
        .into_iter()
        .map(|it| to_proto::inlay_int(&line_index, it))
        .collect())
}

pub fn handle_call_hierarchy_prepare(
    world: WorldSnapshot,
    params: CallHierarchyPrepareParams,
) -> Result<Option<Vec<CallHierarchyItem>>> {
    let _p = profile("handle_call_hierarchy_prepare");
    let position = from_proto::file_position(&world, params.text_document_position_params)?;

    let nav_info = match world.analysis().call_hierarchy(position)? {
        None => return Ok(None),
        Some(it) => it,
    };

    let RangeInfo { range: _, info: navs } = nav_info;
    let res = navs
        .into_iter()
        .filter(|it| it.kind() == SyntaxKind::FN_DEF)
        .map(|it| to_proto::call_hierarchy_item(&world, it))
        .collect::<Result<Vec<_>>>()?;

    Ok(Some(res))
}

pub fn handle_call_hierarchy_incoming(
    world: WorldSnapshot,
    params: CallHierarchyIncomingCallsParams,
) -> Result<Option<Vec<CallHierarchyIncomingCall>>> {
    let _p = profile("handle_call_hierarchy_incoming");
    let item = params.item;

    let doc = TextDocumentIdentifier::new(item.uri);
    let frange = from_proto::file_range(&world, doc, item.range)?;
    let fpos = FilePosition { file_id: frange.file_id, offset: frange.range.start() };

    let call_items = match world.analysis().incoming_calls(fpos)? {
        None => return Ok(None),
        Some(it) => it,
    };

    let mut res = vec![];

    for call_item in call_items.into_iter() {
        let file_id = call_item.target.file_id();
        let line_index = world.analysis().file_line_index(file_id)?;
        let item = to_proto::call_hierarchy_item(&world, call_item.target)?;
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

pub fn handle_call_hierarchy_outgoing(
    world: WorldSnapshot,
    params: CallHierarchyOutgoingCallsParams,
) -> Result<Option<Vec<CallHierarchyOutgoingCall>>> {
    let _p = profile("handle_call_hierarchy_outgoing");
    let item = params.item;

    let doc = TextDocumentIdentifier::new(item.uri);
    let frange = from_proto::file_range(&world, doc, item.range)?;
    let fpos = FilePosition { file_id: frange.file_id, offset: frange.range.start() };

    let call_items = match world.analysis().outgoing_calls(fpos)? {
        None => return Ok(None),
        Some(it) => it,
    };

    let mut res = vec![];

    for call_item in call_items.into_iter() {
        let file_id = call_item.target.file_id();
        let line_index = world.analysis().file_line_index(file_id)?;
        let item = to_proto::call_hierarchy_item(&world, call_item.target)?;
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

pub fn handle_semantic_tokens(
    world: WorldSnapshot,
    params: SemanticTokensParams,
) -> Result<Option<SemanticTokensResult>> {
    let _p = profile("handle_semantic_tokens");

    let file_id = from_proto::file_id(&world, &params.text_document.uri)?;
    let text = world.analysis().file_text(file_id)?;
    let line_index = world.analysis().file_line_index(file_id)?;

    let highlights = world.analysis().highlight(file_id)?;
    let semantic_tokens = to_proto::semantic_tokens(&text, &line_index, highlights);
    Ok(Some(semantic_tokens.into()))
}

pub fn handle_semantic_tokens_range(
    world: WorldSnapshot,
    params: SemanticTokensRangeParams,
) -> Result<Option<SemanticTokensRangeResult>> {
    let _p = profile("handle_semantic_tokens_range");

    let frange = from_proto::file_range(&world, params.text_document, params.range)?;
    let text = world.analysis().file_text(frange.file_id)?;
    let line_index = world.analysis().file_line_index(frange.file_id)?;

    let highlights = world.analysis().highlight_range(frange)?;
    let semantic_tokens = to_proto::semantic_tokens(&text, &line_index, highlights);
    Ok(Some(semantic_tokens.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    use mbe::{ast_to_token_tree, TokenMap};
    use ra_cfg::parse_cfg;
    use ra_syntax::{
        ast::{self, AstNode},
        SmolStr,
    };

    fn get_token_tree_generated(input: &str) -> (tt::Subtree, TokenMap) {
        let source_file = ast::SourceFile::parse(input).ok().unwrap();
        let tt = source_file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
        ast_to_token_tree(&tt).unwrap()
    }

    #[test]
    fn test_cfg_expr_minimal_features_needed() {
        let (subtree, _) = get_token_tree_generated(r#"#![cfg(feature = "baz")]"#);
        let cfg_expr = parse_cfg(&subtree);
        let mut min_features = vec![];
        collect_minimal_features_needed(&cfg_expr, &mut min_features);

        assert_eq!(min_features, vec![SmolStr::new("baz")]);

        let (subtree, _) =
            get_token_tree_generated(r#"#![cfg(all(feature = "baz", feature = "foo"))]"#);
        let cfg_expr = parse_cfg(&subtree);

        let mut min_features = vec![];
        collect_minimal_features_needed(&cfg_expr, &mut min_features);
        assert_eq!(min_features, vec![SmolStr::new("baz"), SmolStr::new("foo")]);

        let (subtree, _) =
            get_token_tree_generated(r#"#![cfg(any(feature = "baz", feature = "foo", unix))]"#);
        let cfg_expr = parse_cfg(&subtree);

        let mut min_features = vec![];
        collect_minimal_features_needed(&cfg_expr, &mut min_features);
        assert_eq!(min_features, vec![SmolStr::new("baz")]);

        let (subtree, _) = get_token_tree_generated(r#"#![cfg(foo)]"#);
        let cfg_expr = parse_cfg(&subtree);

        let mut min_features = vec![];
        collect_minimal_features_needed(&cfg_expr, &mut min_features);
        assert!(min_features.is_empty());
    }
}
