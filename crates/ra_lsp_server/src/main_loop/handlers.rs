use std::{fmt::Write as _, io::Write as _};

use gen_lsp_server::ErrorCode;
use lsp_types::{
    CodeAction, CodeActionResponse, CodeLens, Command, CompletionItem, Diagnostic,
    DocumentFormattingParams, DocumentHighlight, DocumentSymbol, FoldingRange, FoldingRangeKind,
    FoldingRangeParams, Hover, HoverContents, Location, MarkupContent, MarkupKind, Position,
    PrepareRenameResponse, Range, RenameParams, SymbolInformation, TextDocumentIdentifier,
    TextEdit, WorkspaceEdit,
};
use ra_ide_api::{
    AssistId, Cancelable, FileId, FilePosition, FileRange, FoldKind, Query, RunnableKind,
};
use ra_prof::profile;
use ra_syntax::{AstNode, SyntaxKind, TextRange, TextUnit};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::to_value;
use url_serde::Ser;

use crate::{
    cargo_target_spec::{runnable_args, CargoTargetSpec},
    conv::{to_location, Conv, ConvWith, MapConvWith, TryConvWith, TryConvWithToVec},
    req::{self, Decoration, InlayHint, InlayHintsParams, InlayKind},
    world::WorldSnapshot,
    LspError, Result,
};

pub fn handle_analyzer_status(world: WorldSnapshot, _: ()) -> Result<String> {
    let mut buf = world.status();
    writeln!(buf, "\n\nrequests:").unwrap();
    let requests = world.latest_requests.read();
    for (is_last, r) in requests.iter() {
        let mark = if is_last { "*" } else { " " };
        writeln!(buf, "{}{:4} {:<36}{}ms", mark, r.id, r.method, r.duration.as_millis()).unwrap();
    }
    Ok(buf)
}

pub fn handle_syntax_tree(world: WorldSnapshot, params: req::SyntaxTreeParams) -> Result<String> {
    let id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(id)?;
    let text_range = params.range.map(|p| p.conv_with(&line_index));
    let res = world.analysis().syntax_tree(id, text_range)?;
    Ok(res)
}

// FIXME: drop this API
pub fn handle_extend_selection(
    world: WorldSnapshot,
    params: req::ExtendSelectionParams,
) -> Result<req::ExtendSelectionResult> {
    log::error!(
        "extend selection is deprecated and will be removed soon,
         use the new selection range API in LSP",
    );
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let selections = params
        .selections
        .into_iter()
        .map_conv_with(&line_index)
        .map(|range| FileRange { file_id, range })
        .map(|frange| world.analysis().extend_selection(frange).map(|it| it.conv_with(&line_index)))
        .collect::<Cancelable<Vec<_>>>()?;
    Ok(req::ExtendSelectionResult { selections })
}

pub fn handle_selection_range(
    world: WorldSnapshot,
    params: req::SelectionRangeParams,
) -> Result<Vec<req::SelectionRange>> {
    let _p = profile("handle_selection_range");
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    params
        .positions
        .into_iter()
        .map_conv_with(&line_index)
        .map(|position| {
            let mut ranges = Vec::new();
            {
                let mut range = TextRange::from_to(position, position);
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
        .collect()
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

pub fn handle_on_type_formatting(
    world: WorldSnapshot,
    params: req::DocumentOnTypeFormattingParams,
) -> Result<Option<Vec<TextEdit>>> {
    let _p = profile("handle_on_type_formatting");
    let mut position = params.text_document_position.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(position.file_id)?;

    // in `ra_ide_api`, the `on_type` invariant is that
    // `text.char_at(position) == typed_char`.
    position.offset = position.offset - TextUnit::of_char('.');

    let edit = match params.ch.as_str() {
        "=" => world.analysis().on_eq_typed(position),
        "." => world.analysis().on_dot_typed(position),
        _ => return Ok(None),
    }?;
    let mut edit = match edit {
        Some(it) => it,
        None => return Ok(None),
    };

    // This should be a single-file edit
    let edit = edit.source_file_edits.pop().unwrap();

    let change: Vec<TextEdit> = edit.edit.conv_with(&line_index);
    Ok(Some(change))
}

pub fn handle_document_symbol(
    world: WorldSnapshot,
    params: req::DocumentSymbolParams,
) -> Result<Option<req::DocumentSymbolResponse>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;

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
    let mut res = Vec::new();
    while let Some((node, parent)) = parents.pop() {
        match parent {
            None => res.push(node),
            Some(i) => {
                let children = &mut parents[i].0.children;
                if children.is_none() {
                    *children = Some(Vec::new());
                }
                children.as_mut().unwrap().push(node);
            }
        }
    }

    Ok(Some(res.into()))
}

pub fn handle_workspace_symbol(
    world: WorldSnapshot,
    params: req::WorkspaceSymbolParams,
) -> Result<Option<Vec<SymbolInformation>>> {
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
    params: req::TextDocumentPositionParams,
) -> Result<Option<req::GotoDefinitionResponse>> {
    let position = params.try_conv_with(&world)?;
    let nav_info = match world.analysis().goto_definition(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let res = (position.file_id, nav_info).try_conv_with(&world)?;
    Ok(Some(res))
}

pub fn handle_goto_implementation(
    world: WorldSnapshot,
    params: req::TextDocumentPositionParams,
) -> Result<Option<req::GotoImplementationResponse>> {
    let position = params.try_conv_with(&world)?;
    let nav_info = match world.analysis().goto_implementation(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let res = (position.file_id, nav_info).try_conv_with(&world)?;
    Ok(Some(res))
}

pub fn handle_goto_type_definition(
    world: WorldSnapshot,
    params: req::TextDocumentPositionParams,
) -> Result<Option<req::GotoTypeDefinitionResponse>> {
    let position = params.try_conv_with(&world)?;
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
    let position = params.try_conv_with(&world)?;
    world.analysis().parent_module(position)?.iter().try_conv_with_to_vec(&world)
}

pub fn handle_runnables(
    world: WorldSnapshot,
    params: req::RunnablesParams,
) -> Result<Vec<req::Runnable>> {
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

        let args = runnable_args(&world, file_id, &runnable.kind)?;

        let r = req::Runnable {
            range: runnable.range.conv_with(&line_index),
            label: match &runnable.kind {
                RunnableKind::Test { name } => format!("test {}", name),
                RunnableKind::TestMod { path } => format!("test-mod {}", path),
                RunnableKind::Bench { name } => format!("bench {}", name),
                RunnableKind::Bin => "run binary".to_string(),
            },
            bin: "cargo".to_string(),
            args,
            env: {
                let mut m = FxHashMap::default();
                m.insert("RUST_BACKTRACE".to_string(), "short".to_string());
                m
            },
            cwd: workspace_root.map(|root| root.to_string_lossy().to_string()),
        };
        res.push(r);
    }
    let mut check_args = vec!["check".to_string()];
    let label;
    match CargoTargetSpec::for_file(&world, file_id)? {
        Some(spec) => {
            label = format!("cargo check -p {}", spec.package);
            spec.push_to(&mut check_args);
        }
        None => {
            label = "cargo check --all".to_string();
            check_args.push("--all".to_string())
        }
    }
    // Always add `cargo check`.
    res.push(req::Runnable {
        range: Default::default(),
        label,
        bin: "cargo".to_string(),
        args: check_args,
        env: FxHashMap::default(),
        cwd: workspace_root.map(|root| root.to_string_lossy().to_string()),
    });
    Ok(res)
}

pub fn handle_decorations(
    world: WorldSnapshot,
    params: TextDocumentIdentifier,
) -> Result<Vec<Decoration>> {
    let file_id = params.try_conv_with(&world)?;
    highlight(&world, file_id)
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
                    let diff = TextUnit::of_char(next_char) + TextUnit::of_char(':');
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

    let items = match world.analysis().completions(position)? {
        None => return Ok(None),
        Some(items) => items,
    };
    let line_index = world.analysis().file_line_index(position.file_id)?;
    let items: Vec<CompletionItem> =
        items.into_iter().map(|item| item.conv_with(&line_index)).collect();

    Ok(Some(items.into()))
}

pub fn handle_folding_range(
    world: WorldSnapshot,
    params: FoldingRangeParams,
) -> Result<Option<Vec<FoldingRange>>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;

    let res = Some(
        world
            .analysis()
            .folding_ranges(file_id)?
            .into_iter()
            .map(|fold| {
                let kind = match fold.kind {
                    FoldKind::Comment => Some(FoldingRangeKind::Comment),
                    FoldKind::Imports => Some(FoldingRangeKind::Imports),
                    FoldKind::Mods => None,
                    FoldKind::Block => None,
                };
                let range = fold.range.conv_with(&line_index);
                FoldingRange {
                    start_line: range.start.line,
                    start_character: Some(range.start.character),
                    end_line: range.end.line,
                    end_character: Some(range.start.character),
                    kind,
                }
            })
            .collect(),
    );

    Ok(res)
}

pub fn handle_signature_help(
    world: WorldSnapshot,
    params: req::TextDocumentPositionParams,
) -> Result<Option<req::SignatureHelp>> {
    let position = params.try_conv_with(&world)?;
    if let Some(call_info) = world.analysis().call_info(position)? {
        let active_parameter = call_info.active_parameter.map(|it| it as i64);
        let sig_info = call_info.signature.conv();

        Ok(Some(req::SignatureHelp {
            signatures: vec![sig_info],
            active_signature: Some(0),
            active_parameter,
        }))
    } else {
        Ok(None)
    }
}

pub fn handle_hover(
    world: WorldSnapshot,
    params: req::TextDocumentPositionParams,
) -> Result<Option<Hover>> {
    let position = params.try_conv_with(&world)?;
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
    let position = params.try_conv_with(&world)?;

    // We support renaming references like handle_rename does.
    // In the future we may want to reject the renaming of things like keywords here too.
    let refs = match world.analysis().find_all_refs(position)? {
        None => return Ok(None),
        Some(refs) => refs,
    };

    // Refs should always have a declaration
    let r = refs.declaration();
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let loc = to_location(r.file_id(), r.range(), &world, &line_index)?;

    Ok(Some(PrepareRenameResponse::Range(loc.range)))
}

pub fn handle_rename(world: WorldSnapshot, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
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
        Some(it) => it,
    };

    let source_change_req = change.try_conv_with(&world)?;

    Ok(Some(source_change_req.workspace_edit))
}

pub fn handle_references(
    world: WorldSnapshot,
    params: req::ReferenceParams,
) -> Result<Option<Vec<Location>>> {
    let position = params.text_document_position.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(position.file_id)?;

    let refs = match world.analysis().find_all_refs(position)? {
        None => return Ok(None),
        Some(refs) => refs,
    };

    let locations = if params.context.include_declaration {
        refs.into_iter()
            .filter_map(|r| to_location(r.file_id, r.range, &world, &line_index).ok())
            .collect()
    } else {
        // Only iterate over the references if include_declaration was false
        refs.references()
            .iter()
            .filter_map(|r| to_location(r.file_id, r.range, &world, &line_index).ok())
            .collect()
    };

    Ok(Some(locations))
}

pub fn handle_formatting(
    world: WorldSnapshot,
    params: DocumentFormattingParams,
) -> Result<Option<Vec<TextEdit>>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_text(file_id)?;

    let file_line_index = world.analysis().file_line_index(file_id)?;
    let end_position = TextUnit::of_str(&file).conv_with(&file_line_index);

    use std::process;
    let mut rustfmt = process::Command::new("rustfmt");
    rustfmt.stdin(process::Stdio::piped()).stdout(process::Stdio::piped());

    if let Ok(path) = params.text_document.uri.to_file_path() {
        if let Some(parent) = path.parent() {
            rustfmt.current_dir(parent);
        }
    }
    let mut rustfmt = rustfmt.spawn()?;

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

pub fn handle_code_action(
    world: WorldSnapshot,
    params: req::CodeActionParams,
) -> Result<Option<CodeActionResponse>> {
    let _p = profile("handle_code_action");
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let range = params.range.conv_with(&line_index);

    let assists = world.analysis().assists(FileRange { file_id, range })?.into_iter();
    let diagnostics = world.analysis().diagnostics(file_id)?;
    let mut res = CodeActionResponse::default();

    let fixes_from_diagnostics = diagnostics
        .into_iter()
        .filter_map(|d| Some((d.range, d.fix?)))
        .filter(|(diag_range, _fix)| diag_range.intersection(&range).is_some())
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
        };
        res.push(action.into());
    }

    for assist in assists {
        let title = assist.change.label.clone();
        let edit = assist.change.try_conv_with(&world)?;

        let command = Command {
            title,
            command: "rust-analyzer.applySourceChange".to_string(),
            arguments: Some(vec![to_value(edit).unwrap()]),
        };
        let action = CodeAction {
            title: command.title.clone(),
            kind: match assist.id {
                AssistId("introduce_variable") => Some("refactor.extract.variable".to_string()),
                _ => None,
            },
            diagnostics: None,
            edit: None,
            command: Some(command),
        };
        res.push(action.into());
    }

    Ok(Some(res))
}

pub fn handle_code_lens(
    world: WorldSnapshot,
    params: req::CodeLensParams,
) -> Result<Option<Vec<CodeLens>>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;

    let mut lenses: Vec<CodeLens> = Default::default();
    let workspace_root = world.workspace_root_for(file_id);

    // Gather runnables
    for runnable in world.analysis().runnables(file_id)? {
        let title = match &runnable.kind {
            RunnableKind::Test { .. } | RunnableKind::TestMod { .. } => Some("▶️Run Test"),
            RunnableKind::Bench { .. } => Some("Run Bench"),
            RunnableKind::Bin => Some("️Run"),
        };

        if let Some(title) = title {
            let args = runnable_args(&world, file_id, &runnable.kind)?;
            let range = runnable.range.conv_with(&line_index);

            // This represents the actual command that will be run.
            let r: req::Runnable = req::Runnable {
                range,
                label: Default::default(),
                bin: "cargo".into(),
                args,
                env: Default::default(),
                cwd: workspace_root.map(|root| root.to_string_lossy().to_string()),
            };

            let lens = CodeLens {
                range,
                command: Some(Command {
                    title: title.into(),
                    command: "rust-analyzer.runSingle".into(),
                    arguments: Some(vec![to_value(r).unwrap()]),
                }),
                data: None,
            };

            lenses.push(lens);
        }
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
                let lens_params =
                    req::TextDocumentPositionParams::new(params.text_document.clone(), pos);
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
    Impls(req::TextDocumentPositionParams),
}

pub fn handle_code_lens_resolve(world: WorldSnapshot, code_lens: CodeLens) -> Result<CodeLens> {
    let data = code_lens.data.unwrap();
    let resolve = serde_json::from_value(data)?;
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
                    to_value(&Ser::new(&lens_params.text_document.uri)).unwrap(),
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
    params: req::TextDocumentPositionParams,
) -> Result<Option<Vec<DocumentHighlight>>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;

    let refs = match world.analysis().find_all_refs(params.try_conv_with(&world)?)? {
        None => return Ok(None),
        Some(refs) => refs,
    };

    Ok(Some(
        refs.into_iter()
            .map(|r| DocumentHighlight { range: r.range.conv_with(&line_index), kind: None })
            .collect(),
    ))
}

pub fn publish_diagnostics(
    world: &WorldSnapshot,
    file_id: FileId,
) -> Result<req::PublishDiagnosticsParams> {
    let uri = world.file_id_to_uri(file_id)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let diagnostics = world
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
        })
        .collect();
    Ok(req::PublishDiagnosticsParams { uri, diagnostics })
}

pub fn publish_decorations(
    world: &WorldSnapshot,
    file_id: FileId,
) -> Result<req::PublishDecorationsParams> {
    let uri = world.file_id_to_uri(file_id)?;
    Ok(req::PublishDecorationsParams { uri, decorations: highlight(&world, file_id)? })
}

fn highlight(world: &WorldSnapshot, file_id: FileId) -> Result<Vec<Decoration>> {
    let line_index = world.analysis().file_line_index(file_id)?;
    let res = world
        .analysis()
        .highlight(file_id)?
        .into_iter()
        .map(|h| Decoration {
            range: h.range.conv_with(&line_index),
            tag: h.tag,
            binding_hash: h.binding_hash.map(|x| x.to_string()),
        })
        .collect();
    Ok(res)
}

pub fn handle_inlay_hints(
    world: WorldSnapshot,
    params: InlayHintsParams,
) -> Result<Vec<InlayHint>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let analysis = world.analysis();
    let line_index = analysis.file_line_index(file_id)?;
    Ok(analysis
        .inlay_hints(file_id)?
        .into_iter()
        .map(|api_type| InlayHint {
            label: api_type.label.to_string(),
            range: api_type.range.conv_with(&line_index),
            kind: match api_type.kind {
                ra_ide_api::InlayKind::TypeHint => InlayKind::TypeHint,
            },
        })
        .collect())
}
