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
    CodeAction, CodeActionResponse, CodeLens, Command, CompletionItem, Diagnostic,
    DocumentFormattingParams, DocumentHighlight, DocumentSymbol, FoldingRange, FoldingRangeParams,
    Hover, HoverContents, Location, MarkupContent, MarkupKind, Position, PrepareRenameResponse,
    Range, RenameParams, SemanticTokensParams, SemanticTokensRangeParams,
    SemanticTokensRangeResult, SemanticTokensResult, SymbolInformation, TextDocumentIdentifier,
    TextEdit, Url, WorkspaceEdit,
};
use ra_ide::{
    Assist, FileId, FilePosition, FileRange, Query, RangeInfo, Runnable, RunnableKind, SearchScope,
};
use ra_prof::profile;
use ra_syntax::{AstNode, SyntaxKind, TextRange, TextSize};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::to_value;
use stdx::format_to;

use crate::{
    cargo_target_spec::CargoTargetSpec,
    config::RustfmtConfig,
    conv::{
        to_call_hierarchy_item, to_location, Conv, ConvWith, FoldConvCtx, MapConvWith, TryConvWith,
        TryConvWithToVec,
    },
    diagnostics::DiagnosticTask,
    from_json,
    req::{self, InlayHint, InlayHintsParams},
    semantic_tokens::SemanticTokensBuilder,
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

pub fn handle_syntax_tree(world: WorldSnapshot, params: req::SyntaxTreeParams) -> Result<String> {
    let _p = profile("handle_syntax_tree");
    let id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(id)?;
    let text_range = params.range.map(|p| p.conv_with(&line_index));
    let res = world.analysis().syntax_tree(id, text_range)?;
    Ok(res)
}

pub fn handle_expand_macro(
    world: WorldSnapshot,
    params: req::ExpandMacroParams,
) -> Result<Option<req::ExpandedMacro>> {
    let _p = profile("handle_expand_macro");
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let offset = params.position.map(|p| p.conv_with(&line_index));

    match offset {
        None => Ok(None),
        Some(offset) => {
            let res = world.analysis().expand_macro(FilePosition { file_id, offset })?;
            Ok(res.map(|it| req::ExpandedMacro { name: it.name, expansion: it.expansion }))
        }
    }
}

pub fn handle_selection_range(
    world: WorldSnapshot,
    params: req::SelectionRangeParams,
) -> Result<Option<Vec<req::SelectionRange>>> {
    let _p = profile("handle_selection_range");
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let res: Result<Vec<req::SelectionRange>> = params
        .positions
        .into_iter()
        .map_conv_with(&line_index)
        .map(|position| {
            let mut ranges = Vec::new();
            {
                let mut range = TextRange::new(position, position);
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
            let mut range = req::SelectionRange {
                range: ranges.last().unwrap().conv_with(&line_index),
                parent: None,
            };
            for r in ranges.iter().rev().skip(1) {
                range = req::SelectionRange {
                    range: r.conv_with(&line_index),
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
    params: req::FindMatchingBraceParams,
) -> Result<Vec<Position>> {
    let _p = profile("handle_find_matching_brace");
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let res = params
        .offsets
        .into_iter()
        .map_conv_with(&line_index)
        .map(|offset| {
            if let Ok(Some(matching_brace_offset)) =
                world.analysis().matching_brace(FilePosition { file_id, offset })
            {
                matching_brace_offset
            } else {
                offset
            }
        })
        .map_conv_with(&line_index)
        .collect();
    Ok(res)
}

pub fn handle_join_lines(
    world: WorldSnapshot,
    params: req::JoinLinesParams,
) -> Result<req::SourceChange> {
    let _p = profile("handle_join_lines");
    let frange = (&params.text_document, params.range).try_conv_with(&world)?;
    world.analysis().join_lines(frange)?.try_conv_with(&world)
}

pub fn handle_on_enter(
    world: WorldSnapshot,
    params: req::TextDocumentPositionParams,
) -> Result<Option<req::SourceChange>> {
    let _p = profile("handle_on_enter");
    let position = params.try_conv_with(&world)?;
    match world.analysis().on_enter(position)? {
        None => Ok(None),
        Some(edit) => Ok(Some(edit.try_conv_with(&world)?)),
    }
}

// Don't forget to add new trigger characters to `ServerCapabilities` in `caps.rs`.
pub fn handle_on_type_formatting(
    world: WorldSnapshot,
    params: req::DocumentOnTypeFormattingParams,
) -> Result<Option<Vec<TextEdit>>> {
    let _p = profile("handle_on_type_formatting");
    let mut position = params.text_document_position.try_conv_with(&world)?;
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

    let change: Vec<TextEdit> = edit.edit.conv_with((&line_index, line_endings));
    Ok(Some(change))
}

pub fn handle_document_symbol(
    world: WorldSnapshot,
    params: req::DocumentSymbolParams,
) -> Result<Option<req::DocumentSymbolResponse>> {
    let _p = profile("handle_document_symbol");
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let url = file_id.try_conv_with(&world)?;

    let mut parents: Vec<(DocumentSymbol, Option<usize>)> = Vec::new();

    for symbol in world.analysis().file_structure(file_id)? {
        let doc_symbol = DocumentSymbol {
            name: symbol.label,
            detail: symbol.detail,
            kind: symbol.kind.conv(),
            deprecated: Some(symbol.deprecated),
            range: symbol.node_range.conv_with(&line_index),
            selection_range: symbol.navigation_range.conv_with(&line_index),
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

    if world.config.client_caps.hierarchical_symbols {
        Ok(Some(document_symbols.into()))
    } else {
        let mut symbol_information = Vec::<SymbolInformation>::new();
        for symbol in document_symbols {
            flatten_document_symbol(&symbol, None, &url, &mut symbol_information);
        }

        Ok(Some(symbol_information.into()))
    }
}

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
        container_name: container_name,
    });

    for child in symbol.children.iter().flatten() {
        flatten_document_symbol(child, Some(symbol.name.clone()), url, res);
    }
}

pub fn handle_workspace_symbol(
    world: WorldSnapshot,
    params: req::WorkspaceSymbolParams,
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
                kind: nav.kind().conv(),
                location: nav.try_conv_with(world)?,
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
    params: req::GotoDefinitionParams,
) -> Result<Option<req::GotoDefinitionResponse>> {
    let _p = profile("handle_goto_definition");
    let position = params.text_document_position_params.try_conv_with(&world)?;
    let nav_info = match world.analysis().goto_definition(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let res = (position.file_id, nav_info).try_conv_with(&world)?;
    Ok(Some(res))
}

pub fn handle_goto_implementation(
    world: WorldSnapshot,
    params: req::GotoImplementationParams,
) -> Result<Option<req::GotoImplementationResponse>> {
    let _p = profile("handle_goto_implementation");
    let position = params.text_document_position_params.try_conv_with(&world)?;
    let nav_info = match world.analysis().goto_implementation(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let res = (position.file_id, nav_info).try_conv_with(&world)?;
    Ok(Some(res))
}

pub fn handle_goto_type_definition(
    world: WorldSnapshot,
    params: req::GotoTypeDefinitionParams,
) -> Result<Option<req::GotoTypeDefinitionResponse>> {
    let _p = profile("handle_goto_type_definition");
    let position = params.text_document_position_params.try_conv_with(&world)?;
    let nav_info = match world.analysis().goto_type_definition(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let res = (position.file_id, nav_info).try_conv_with(&world)?;
    Ok(Some(res))
}

pub fn handle_parent_module(
    world: WorldSnapshot,
    params: req::TextDocumentPositionParams,
) -> Result<Vec<Location>> {
    let _p = profile("handle_parent_module");
    let position = params.try_conv_with(&world)?;
    world.analysis().parent_module(position)?.iter().try_conv_with_to_vec(&world)
}

pub fn handle_runnables(
    world: WorldSnapshot,
    params: req::RunnablesParams,
) -> Result<Vec<req::Runnable>> {
    let _p = profile("handle_runnables");
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let offset = params.position.map(|it| it.conv_with(&line_index));
    let mut res = Vec::new();
    let workspace_root = world.workspace_root_for(file_id);
    for runnable in world.analysis().runnables(file_id)? {
        if let Some(offset) = offset {
            if !runnable.range.contains_inclusive(offset) {
                continue;
            }
        }
        res.push(to_lsp_runnable(&world, file_id, runnable)?);
    }
    // Add `cargo check` and `cargo test` for the whole package
    match CargoTargetSpec::for_file(&world, file_id)? {
        Some(spec) => {
            for &cmd in ["check", "test"].iter() {
                res.push(req::Runnable {
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
            res.push(req::Runnable {
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
    params: req::CompletionParams,
) -> Result<Option<req::CompletionResponse>> {
    let _p = profile("handle_completion");
    let position = params.text_document_position.try_conv_with(&world)?;
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

    let items = match world.analysis().completions(position, &world.config.completion)? {
        None => return Ok(None),
        Some(items) => items,
    };
    let line_index = world.analysis().file_line_index(position.file_id)?;
    let line_endings = world.file_line_endings(position.file_id);
    let items: Vec<CompletionItem> =
        items.into_iter().map(|item| item.conv_with((&line_index, line_endings))).collect();

    Ok(Some(items.into()))
}

pub fn handle_folding_range(
    world: WorldSnapshot,
    params: FoldingRangeParams,
) -> Result<Option<Vec<FoldingRange>>> {
    let _p = profile("handle_folding_range");
    let file_id = params.text_document.try_conv_with(&world)?;
    let folds = world.analysis().folding_ranges(file_id)?;
    let text = world.analysis().file_text(file_id)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let ctx = FoldConvCtx {
        text: &text,
        line_index: &line_index,
        line_folding_only: world.config.client_caps.line_folding_only,
    };
    let res = Some(folds.into_iter().map_conv_with(&ctx).collect());
    Ok(res)
}

pub fn handle_signature_help(
    world: WorldSnapshot,
    params: req::SignatureHelpParams,
) -> Result<Option<req::SignatureHelp>> {
    let _p = profile("handle_signature_help");
    let position = params.text_document_position_params.try_conv_with(&world)?;
    if let Some(call_info) = world.analysis().call_info(position)? {
        let concise = !world.config.call_info_full;
        let mut active_parameter = call_info.active_parameter.map(|it| it as i64);
        if concise && call_info.signature.has_self_param {
            active_parameter = active_parameter.map(|it| it.saturating_sub(1));
        }
        let sig_info = call_info.signature.conv_with(concise);

        Ok(Some(req::SignatureHelp {
            signatures: vec![sig_info],
            active_signature: Some(0),
            active_parameter,
        }))
    } else {
        Ok(None)
    }
}

pub fn handle_hover(world: WorldSnapshot, params: req::HoverParams) -> Result<Option<Hover>> {
    let _p = profile("handle_hover");
    let position = params.text_document_position_params.try_conv_with(&world)?;
    let info = match world.analysis().hover(position)? {
        None => return Ok(None),
        Some(info) => info,
    };
    let line_index = world.analysis.file_line_index(position.file_id)?;
    let range = info.range.conv_with(&line_index);
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
    params: req::TextDocumentPositionParams,
) -> Result<Option<PrepareRenameResponse>> {
    let _p = profile("handle_prepare_rename");
    let position = params.try_conv_with(&world)?;

    let optional_change = world.analysis().rename(position, "dummy")?;
    let range = match optional_change {
        None => return Ok(None),
        Some(it) => it.range,
    };

    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let range = range.conv_with(&line_index);
    Ok(Some(PrepareRenameResponse::Range(range)))
}

pub fn handle_rename(world: WorldSnapshot, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
    let _p = profile("handle_rename");
    let position = params.text_document_position.try_conv_with(&world)?;

    if params.new_name.is_empty() {
        return Err(LspError::new(
            ErrorCode::InvalidParams as i32,
            "New Name cannot be empty".into(),
        )
        .into());
    }

    let optional_change = world.analysis().rename(position, &*params.new_name)?;
    let change = match optional_change {
        None => return Ok(None),
        Some(it) => it.info,
    };

    let source_change_req = change.try_conv_with(&world)?;

    Ok(Some(source_change_req.workspace_edit))
}

pub fn handle_references(
    world: WorldSnapshot,
    params: req::ReferenceParams,
) -> Result<Option<Vec<Location>>> {
    let _p = profile("handle_references");
    let position = params.text_document_position.try_conv_with(&world)?;

    let refs = match world.analysis().find_all_refs(position, None)? {
        None => return Ok(None),
        Some(refs) => refs,
    };

    let locations = if params.context.include_declaration {
        refs.into_iter()
            .filter_map(|reference| {
                let line_index =
                    world.analysis().file_line_index(reference.file_range.file_id).ok()?;
                to_location(
                    reference.file_range.file_id,
                    reference.file_range.range,
                    &world,
                    &line_index,
                )
                .ok()
            })
            .collect()
    } else {
        // Only iterate over the references if include_declaration was false
        refs.references()
            .iter()
            .filter_map(|reference| {
                let line_index =
                    world.analysis().file_line_index(reference.file_range.file_id).ok()?;
                to_location(
                    reference.file_range.file_id,
                    reference.file_range.range,
                    &world,
                    &line_index,
                )
                .ok()
            })
            .collect()
    };

    Ok(Some(locations))
}

pub fn handle_formatting(
    world: WorldSnapshot,
    params: DocumentFormattingParams,
) -> Result<Option<Vec<TextEdit>>> {
    let _p = profile("handle_formatting");
    let file_id = params.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_text(file_id)?;
    let crate_ids = world.analysis().crate_for(file_id)?;

    let file_line_index = world.analysis().file_line_index(file_id)?;
    let end_position = TextSize::of(file.as_str()).conv_with(&file_line_index);

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

    Ok(Some(vec![TextEdit {
        range: Range::new(Position::new(0, 0), end_position),
        new_text: captured_stdout,
    }]))
}

fn create_single_code_action(assist: Assist, world: &WorldSnapshot) -> Result<CodeAction> {
    let arg = to_value(assist.source_change.try_conv_with(world)?)?;
    let title = assist.label;
    let command = Command {
        title: title.clone(),
        command: "rust-analyzer.applySourceChange".to_string(),
        arguments: Some(vec![arg]),
    };

    Ok(CodeAction {
        title,
        kind: Some(String::new()),
        diagnostics: None,
        edit: None,
        command: Some(command),
        is_preferred: None,
    })
}

pub fn handle_code_action(
    world: WorldSnapshot,
    params: req::CodeActionParams,
) -> Result<Option<CodeActionResponse>> {
    let _p = profile("handle_code_action");
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let range = params.range.conv_with(&line_index);

    let diagnostics = world.analysis().diagnostics(file_id)?;
    let mut res = CodeActionResponse::default();

    let fixes_from_diagnostics = diagnostics
        .into_iter()
        .filter_map(|d| Some((d.range, d.fix?)))
        .filter(|(diag_range, _fix)| diag_range.intersect(range).is_some())
        .map(|(_range, fix)| fix);

    for source_edit in fixes_from_diagnostics {
        let title = source_edit.label.clone();
        let edit = source_edit.try_conv_with(&world)?;

        let command = Command {
            title,
            command: "rust-analyzer.applySourceChange".to_string(),
            arguments: Some(vec![to_value(edit).unwrap()]),
        };
        let action = CodeAction {
            title: command.title.clone(),
            kind: None,
            diagnostics: None,
            edit: None,
            command: Some(command),
            is_preferred: None,
        };
        res.push(action.into());
    }

    for fix in world.check_fixes.get(&file_id).into_iter().flatten() {
        let fix_range = fix.range.conv_with(&line_index);
        if fix_range.intersect(range).is_none() {
            continue;
        }
        res.push(fix.action.clone());
    }

    let mut grouped_assists: FxHashMap<String, (usize, Vec<Assist>)> = FxHashMap::default();
    for assist in world.analysis().assists(FileRange { file_id, range })?.into_iter() {
        match &assist.group_label {
            Some(label) => grouped_assists
                .entry(label.to_owned())
                .or_insert_with(|| {
                    let idx = res.len();
                    let dummy = Command::new(String::new(), String::new(), None);
                    res.push(dummy.into());
                    (idx, Vec::new())
                })
                .1
                .push(assist),
            None => {
                res.push(create_single_code_action(assist, &world)?.into());
            }
        }
    }

    for (group_label, (idx, assists)) in grouped_assists {
        if assists.len() == 1 {
            res[idx] =
                create_single_code_action(assists.into_iter().next().unwrap(), &world)?.into();
        } else {
            let title = group_label;

            let mut arguments = Vec::with_capacity(assists.len());
            for assist in assists {
                arguments.push(to_value(assist.source_change.try_conv_with(&world)?)?);
            }

            let command = Some(Command {
                title: title.clone(),
                command: "rust-analyzer.selectAndApplySourceChange".to_string(),
                arguments: Some(vec![serde_json::Value::Array(arguments)]),
            });
            res[idx] = CodeAction {
                title,
                kind: None,
                diagnostics: None,
                edit: None,
                command,
                is_preferred: None,
            }
            .into();
        }
    }

    // If the client only supports commands then filter the list
    // and remove and actions that depend on edits.
    if !world.config.client_caps.code_action_literals {
        // FIXME: use drain_filter once it hits stable.
        res = res
            .into_iter()
            .filter_map(|it| match it {
                cmd @ lsp_types::CodeActionOrCommand::Command(_) => Some(cmd),
                lsp_types::CodeActionOrCommand::CodeAction(action) => match action.command {
                    Some(cmd) if action.edit.is_none() => {
                        Some(lsp_types::CodeActionOrCommand::Command(cmd))
                    }
                    _ => None,
                },
            })
            .collect();
    }
    Ok(Some(res))
}

pub fn handle_code_lens(
    world: WorldSnapshot,
    params: req::CodeLensParams,
) -> Result<Option<Vec<CodeLens>>> {
    let _p = profile("handle_code_lens");
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;

    let mut lenses: Vec<CodeLens> = Default::default();

    // Gather runnables
    for runnable in world.analysis().runnables(file_id)? {
        let title = match &runnable.kind {
            RunnableKind::Test { .. } | RunnableKind::TestMod { .. } => "▶️\u{fe0e}Run Test",
            RunnableKind::DocTest { .. } => "▶️\u{fe0e}Run Doctest",
            RunnableKind::Bench { .. } => "Run Bench",
            RunnableKind::Bin => "Run",
        }
        .to_string();
        let mut r = to_lsp_runnable(&world, file_id, runnable)?;
        let lens = CodeLens {
            range: r.range,
            command: Some(Command {
                title,
                command: "rust-analyzer.runSingle".into(),
                arguments: Some(vec![to_value(&r).unwrap()]),
            }),
            data: None,
        };
        lenses.push(lens);

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
                let range = it.node_range.conv_with(&line_index);
                let pos = range.start;
                let lens_params = req::GotoImplementationParams {
                    text_document_position_params: req::TextDocumentPositionParams::new(
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

    Ok(Some(lenses))
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
enum CodeLensResolveData {
    Impls(req::GotoImplementationParams),
}

pub fn handle_code_lens_resolve(world: WorldSnapshot, code_lens: CodeLens) -> Result<CodeLens> {
    let _p = profile("handle_code_lens_resolve");
    let data = code_lens.data.unwrap();
    let resolve = from_json::<Option<CodeLensResolveData>>("CodeLensResolveData", data)?;
    match resolve {
        Some(CodeLensResolveData::Impls(lens_params)) => {
            let locations: Vec<Location> =
                match handle_goto_implementation(world, lens_params.clone())? {
                    Some(req::GotoDefinitionResponse::Scalar(loc)) => vec![loc],
                    Some(req::GotoDefinitionResponse::Array(locs)) => locs,
                    Some(req::GotoDefinitionResponse::Link(links)) => links
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
    params: req::DocumentHighlightParams,
) -> Result<Option<Vec<DocumentHighlight>>> {
    let _p = profile("handle_document_highlight");
    let file_id = params.text_document_position_params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;

    let refs = match world.analysis().find_all_refs(
        params.text_document_position_params.try_conv_with(&world)?,
        Some(SearchScope::single_file(file_id)),
    )? {
        None => return Ok(None),
        Some(refs) => refs,
    };

    Ok(Some(
        refs.into_iter()
            .filter(|reference| reference.file_range.file_id == file_id)
            .map(|reference| DocumentHighlight {
                range: reference.file_range.range.conv_with(&line_index),
                kind: reference.access.map(|it| it.conv()),
            })
            .collect(),
    ))
}

pub fn handle_ssr(world: WorldSnapshot, params: req::SsrParams) -> Result<req::SourceChange> {
    let _p = profile("handle_ssr");
    world
        .analysis()
        .structural_search_replace(&params.query, params.parse_only)??
        .try_conv_with(&world)
}

pub fn publish_diagnostics(world: &WorldSnapshot, file_id: FileId) -> Result<DiagnosticTask> {
    let _p = profile("publish_diagnostics");
    let line_index = world.analysis().file_line_index(file_id)?;
    let diagnostics: Vec<Diagnostic> = world
        .analysis()
        .diagnostics(file_id)?
        .into_iter()
        .map(|d| Diagnostic {
            range: d.range.conv_with(&line_index),
            severity: Some(d.severity.conv()),
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
) -> Result<req::Runnable> {
    let spec = CargoTargetSpec::for_file(world, file_id)?;
    let (args, extra_args) = CargoTargetSpec::runnable_args(spec, &runnable.kind)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let label = match &runnable.kind {
        RunnableKind::Test { test_id, .. } => format!("test {}", test_id),
        RunnableKind::TestMod { path } => format!("test-mod {}", path),
        RunnableKind::Bench { test_id } => format!("bench {}", test_id),
        RunnableKind::DocTest { test_id, .. } => format!("doctest {}", test_id),
        RunnableKind::Bin => "run binary".to_string(),
    };
    Ok(req::Runnable {
        range: runnable.range.conv_with(&line_index),
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

pub fn handle_inlay_hints(
    world: WorldSnapshot,
    params: InlayHintsParams,
) -> Result<Vec<InlayHint>> {
    let _p = profile("handle_inlay_hints");
    let file_id = params.text_document.try_conv_with(&world)?;
    let analysis = world.analysis();
    let line_index = analysis.file_line_index(file_id)?;
    Ok(analysis
        .inlay_hints(file_id, &world.config.inlay_hints)?
        .into_iter()
        .map_conv_with(&line_index)
        .collect())
}

pub fn handle_call_hierarchy_prepare(
    world: WorldSnapshot,
    params: CallHierarchyPrepareParams,
) -> Result<Option<Vec<CallHierarchyItem>>> {
    let _p = profile("handle_call_hierarchy_prepare");
    let position = params.text_document_position_params.try_conv_with(&world)?;
    let file_id = position.file_id;

    let nav_info = match world.analysis().call_hierarchy(position)? {
        None => return Ok(None),
        Some(it) => it,
    };

    let line_index = world.analysis().file_line_index(file_id)?;
    let RangeInfo { range, info: navs } = nav_info;
    let res = navs
        .into_iter()
        .filter(|it| it.kind() == SyntaxKind::FN_DEF)
        .filter_map(|it| to_call_hierarchy_item(file_id, range, &world, &line_index, it).ok())
        .collect();

    Ok(Some(res))
}

pub fn handle_call_hierarchy_incoming(
    world: WorldSnapshot,
    params: CallHierarchyIncomingCallsParams,
) -> Result<Option<Vec<CallHierarchyIncomingCall>>> {
    let _p = profile("handle_call_hierarchy_incoming");
    let item = params.item;

    let doc = TextDocumentIdentifier::new(item.uri);
    let frange: FileRange = (&doc, item.range).try_conv_with(&world)?;
    let fpos = FilePosition { file_id: frange.file_id, offset: frange.range.start() };

    let call_items = match world.analysis().incoming_calls(fpos)? {
        None => return Ok(None),
        Some(it) => it,
    };

    let mut res = vec![];

    for call_item in call_items.into_iter() {
        let file_id = call_item.target.file_id();
        let line_index = world.analysis().file_line_index(file_id)?;
        let range = call_item.target.range();
        let item = to_call_hierarchy_item(file_id, range, &world, &line_index, call_item.target)?;
        res.push(CallHierarchyIncomingCall {
            from: item,
            from_ranges: call_item.ranges.iter().map(|it| it.conv_with(&line_index)).collect(),
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
    let frange: FileRange = (&doc, item.range).try_conv_with(&world)?;
    let fpos = FilePosition { file_id: frange.file_id, offset: frange.range.start() };

    let call_items = match world.analysis().outgoing_calls(fpos)? {
        None => return Ok(None),
        Some(it) => it,
    };

    let mut res = vec![];

    for call_item in call_items.into_iter() {
        let file_id = call_item.target.file_id();
        let line_index = world.analysis().file_line_index(file_id)?;
        let range = call_item.target.range();
        let item = to_call_hierarchy_item(file_id, range, &world, &line_index, call_item.target)?;
        res.push(CallHierarchyOutgoingCall {
            to: item,
            from_ranges: call_item.ranges.iter().map(|it| it.conv_with(&line_index)).collect(),
        });
    }

    Ok(Some(res))
}

pub fn handle_semantic_tokens(
    world: WorldSnapshot,
    params: SemanticTokensParams,
) -> Result<Option<SemanticTokensResult>> {
    let _p = profile("handle_semantic_tokens");

    let file_id = params.text_document.try_conv_with(&world)?;
    let text = world.analysis().file_text(file_id)?;
    let line_index = world.analysis().file_line_index(file_id)?;

    let mut builder = SemanticTokensBuilder::default();

    for highlight_range in world.analysis().highlight(file_id)?.into_iter() {
        let (token_index, modifier_bitset) = highlight_range.highlight.conv();
        for mut range in line_index.lines(highlight_range.range) {
            if text[range].ends_with('\n') {
                range = TextRange::new(range.start(), range.end() - TextSize::of('\n'));
            }
            let range = range.conv_with(&line_index);
            builder.push(range, token_index, modifier_bitset);
        }
    }

    let tokens = builder.build();

    Ok(Some(tokens.into()))
}

pub fn handle_semantic_tokens_range(
    world: WorldSnapshot,
    params: SemanticTokensRangeParams,
) -> Result<Option<SemanticTokensRangeResult>> {
    let _p = profile("handle_semantic_tokens_range");

    let frange = (&params.text_document, params.range).try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(frange.file_id)?;

    let mut builder = SemanticTokensBuilder::default();

    for highlight_range in world.analysis().highlight_range(frange)?.into_iter() {
        let (token_type, token_modifiers) = highlight_range.highlight.conv();
        builder.push(highlight_range.range.conv_with(&line_index), token_type, token_modifiers);
    }

    let tokens = builder.build();

    Ok(Some(tokens.into()))
}
