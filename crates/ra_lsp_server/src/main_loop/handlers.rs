use std::collections::HashMap;

use gen_lsp_server::ErrorCode;
use languageserver_types::{
    CodeActionResponse, Command, Diagnostic,
    DiagnosticSeverity, DocumentSymbol, Documentation, FoldingRange, FoldingRangeKind,
    FoldingRangeParams, Location, MarkupContent, MarkupKind, MarkedString, Position,
    PrepareRenameResponse, RenameParams, SymbolInformation, TextDocumentIdentifier, TextEdit,
    WorkspaceEdit, ParameterInformation, ParameterLabel, SignatureInformation, Hover, HoverContents,
};
use ra_analysis::{FileId, FoldKind, Query, RunnableKind, FilePosition, Severity};
use ra_syntax::{TextUnit, text_utils::intersect};
use ra_text_edit::text_utils::contains_offset_nonstrict;
use rustc_hash::FxHashMap;
use serde_json::to_value;

use crate::{
    conv::{to_location, Conv, ConvWith, MapConvWith, TryConvWith},
    project_model::TargetKind,
    req::{self, Decoration},
    server_world::ServerWorld,
    LspError, Result,
};

pub fn handle_syntax_tree(world: ServerWorld, params: req::SyntaxTreeParams) -> Result<String> {
    let id = params.text_document.try_conv_with(&world)?;
    let res = world.analysis().syntax_tree(id);
    Ok(res)
}

pub fn handle_extend_selection(
    world: ServerWorld,
    params: req::ExtendSelectionParams,
) -> Result<req::ExtendSelectionResult> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_syntax(file_id);
    let line_index = world.analysis().file_line_index(file_id);
    let selections = params
        .selections
        .into_iter()
        .map_conv_with(&line_index)
        .map(|r| world.analysis().extend_selection(&file, r))
        .map_conv_with(&line_index)
        .collect();
    Ok(req::ExtendSelectionResult { selections })
}

pub fn handle_find_matching_brace(
    world: ServerWorld,
    params: req::FindMatchingBraceParams,
) -> Result<Vec<Position>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_syntax(file_id);
    let line_index = world.analysis().file_line_index(file_id);
    let res = params
        .offsets
        .into_iter()
        .map_conv_with(&line_index)
        .map(|offset| {
            world
                .analysis()
                .matching_brace(&file, offset)
                .unwrap_or(offset)
        })
        .map_conv_with(&line_index)
        .collect();
    Ok(res)
}

pub fn handle_join_lines(
    world: ServerWorld,
    params: req::JoinLinesParams,
) -> Result<req::SourceChange> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);
    let range = params.range.conv_with(&line_index);
    world
        .analysis()
        .join_lines(file_id, range)
        .try_conv_with(&world)
}

pub fn handle_on_enter(
    world: ServerWorld,
    params: req::TextDocumentPositionParams,
) -> Result<Option<req::SourceChange>> {
    let position = params.try_conv_with(&world)?;
    match world.analysis().on_enter(position) {
        None => Ok(None),
        Some(edit) => Ok(Some(edit.try_conv_with(&world)?)),
    }
}

pub fn handle_on_type_formatting(
    world: ServerWorld,
    params: req::DocumentOnTypeFormattingParams,
) -> Result<Option<Vec<TextEdit>>> {
    if params.ch != "=" {
        return Ok(None);
    }

    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);
    let position = FilePosition {
        file_id,
        offset: params.position.conv_with(&line_index),
    };
    let edits = match world.analysis().on_eq_typed(position) {
        None => return Ok(None),
        Some(mut action) => action
            .source_file_edits
            .pop()
            .unwrap()
            .edit
            .as_atoms()
            .iter()
            .map_conv_with(&line_index)
            .collect(),
    };
    Ok(Some(edits))
}

pub fn handle_document_symbol(
    world: ServerWorld,
    params: req::DocumentSymbolParams,
) -> Result<Option<req::DocumentSymbolResponse>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);

    let mut parents: Vec<(DocumentSymbol, Option<usize>)> = Vec::new();

    for symbol in world.analysis().file_structure(file_id) {
        let doc_symbol = DocumentSymbol {
            name: symbol.label,
            detail: Some("".to_string()),
            kind: symbol.kind.conv(),
            deprecated: None,
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

    Ok(Some(req::DocumentSymbolResponse::Nested(res)))
}

pub fn handle_workspace_symbol(
    world: ServerWorld,
    params: req::WorkspaceSymbolParams,
) -> Result<Option<Vec<SymbolInformation>>> {
    let all_symbols = params.query.contains('#');
    let libs = params.query.contains('*');
    let query = {
        let query: String = params
            .query
            .chars()
            .filter(|&c| c != '#' && c != '*')
            .collect();
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

    fn exec_query(world: &ServerWorld, query: Query) -> Result<Vec<SymbolInformation>> {
        let mut res = Vec::new();
        for (file_id, symbol) in world.analysis().symbol_search(query)? {
            let line_index = world.analysis().file_line_index(file_id);
            let info = SymbolInformation {
                name: symbol.name.to_string(),
                kind: symbol.kind.conv(),
                location: to_location(file_id, symbol.node_range, world, &line_index)?,
                container_name: None,
                deprecated: None,
            };
            res.push(info);
        }
        Ok(res)
    }
}

pub fn handle_goto_definition(
    world: ServerWorld,
    params: req::TextDocumentPositionParams,
) -> Result<Option<req::GotoDefinitionResponse>> {
    let position = params.try_conv_with(&world)?;
    let rr = match world.analysis().approximately_resolve_symbol(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let mut res = Vec::new();
    for (file_id, symbol) in rr.resolves_to {
        let line_index = world.analysis().file_line_index(file_id);
        let location = to_location(file_id, symbol.node_range, &world, &line_index)?;
        res.push(location)
    }
    Ok(Some(req::GotoDefinitionResponse::Array(res)))
}

pub fn handle_parent_module(
    world: ServerWorld,
    params: req::TextDocumentPositionParams,
) -> Result<Vec<Location>> {
    let position = params.try_conv_with(&world)?;
    let mut res = Vec::new();
    for (file_id, symbol) in world.analysis().parent_module(position)? {
        let line_index = world.analysis().file_line_index(file_id);
        let location = to_location(file_id, symbol.node_range, &world, &line_index)?;
        res.push(location);
    }
    Ok(res)
}

pub fn handle_runnables(
    world: ServerWorld,
    params: req::RunnablesParams,
) -> Result<Vec<req::Runnable>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);
    let offset = params.position.map(|it| it.conv_with(&line_index));
    let mut res = Vec::new();
    for runnable in world.analysis().runnables(file_id)? {
        if let Some(offset) = offset {
            if !contains_offset_nonstrict(runnable.range, offset) {
                continue;
            }
        }

        let args = runnable_args(&world, file_id, &runnable.kind)?;

        let r = req::Runnable {
            range: runnable.range.conv_with(&line_index),
            label: match &runnable.kind {
                RunnableKind::Test { name } => format!("test {}", name),
                RunnableKind::Bin => "run binary".to_string(),
            },
            bin: "cargo".to_string(),
            args,
            env: {
                let mut m = FxHashMap::default();
                m.insert("RUST_BACKTRACE".to_string(), "short".to_string());
                m
            },
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
    });
    return Ok(res);

    fn runnable_args(
        world: &ServerWorld,
        file_id: FileId,
        kind: &RunnableKind,
    ) -> Result<Vec<String>> {
        let spec = CargoTargetSpec::for_file(world, file_id)?;
        let mut res = Vec::new();
        match kind {
            RunnableKind::Test { name } => {
                res.push("test".to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut res);
                }
                res.push("--".to_string());
                res.push(name.to_string());
                res.push("--nocapture".to_string());
            }
            RunnableKind::Bin => {
                res.push("run".to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut res);
                }
            }
        }
        Ok(res)
    }

    struct CargoTargetSpec {
        package: String,
        target: String,
        target_kind: TargetKind,
    }

    impl CargoTargetSpec {
        fn for_file(world: &ServerWorld, file_id: FileId) -> Result<Option<CargoTargetSpec>> {
            let &crate_id = match world.analysis().crate_for(file_id)?.first() {
                Some(crate_id) => crate_id,
                None => return Ok(None),
            };
            let file_id = world.analysis().crate_root(crate_id)?;
            let path = world.vfs.read().file2path(ra_vfs::VfsFile(file_id.0));
            let res = world.workspaces.iter().find_map(|ws| {
                let tgt = ws.target_by_root(&path)?;
                let res = CargoTargetSpec {
                    package: tgt.package(ws).name(ws).to_string(),
                    target: tgt.name(ws).to_string(),
                    target_kind: tgt.kind(ws),
                };
                Some(res)
            });
            Ok(res)
        }

        fn push_to(self, buf: &mut Vec<String>) {
            buf.push("--package".to_string());
            buf.push(self.package);
            match self.target_kind {
                TargetKind::Bin => {
                    buf.push("--bin".to_string());
                    buf.push(self.target);
                }
                TargetKind::Test => {
                    buf.push("--test".to_string());
                    buf.push(self.target);
                }
                TargetKind::Bench => {
                    buf.push("--bench".to_string());
                    buf.push(self.target);
                }
                TargetKind::Example => {
                    buf.push("--example".to_string());
                    buf.push(self.target);
                }
                TargetKind::Lib => {
                    buf.push("--lib".to_string());
                }
                TargetKind::Other => (),
            }
        }
    }
}

pub fn handle_decorations(
    world: ServerWorld,
    params: TextDocumentIdentifier,
) -> Result<Vec<Decoration>> {
    let file_id = params.try_conv_with(&world)?;
    highlight(&world, file_id)
}

pub fn handle_completion(
    world: ServerWorld,
    params: req::CompletionParams,
) -> Result<Option<req::CompletionResponse>> {
    let position = {
        let file_id = params.text_document.try_conv_with(&world)?;
        let line_index = world.analysis().file_line_index(file_id);
        let offset = params.position.conv_with(&line_index);
        FilePosition { file_id, offset }
    };
    let completion_triggered_after_single_colon = {
        let mut res = false;
        if let Some(ctx) = params.context {
            if ctx.trigger_character.unwrap_or_default() == ":" {
                let source_file = world.analysis().file_syntax(position.file_id);
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
    let items = items.into_iter().map(|item| item.conv()).collect();

    Ok(Some(req::CompletionResponse::Array(items)))
}

pub fn handle_folding_range(
    world: ServerWorld,
    params: FoldingRangeParams,
) -> Result<Option<Vec<FoldingRange>>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);

    let res = Some(
        world
            .analysis()
            .folding_ranges(file_id)
            .into_iter()
            .map(|fold| {
                let kind = match fold.kind {
                    FoldKind::Comment => Some(FoldingRangeKind::Comment),
                    FoldKind::Imports => Some(FoldingRangeKind::Imports),
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
    world: ServerWorld,
    params: req::TextDocumentPositionParams,
) -> Result<Option<req::SignatureHelp>> {
    let position = params.try_conv_with(&world)?;

    if let Some((descriptor, active_param)) = world.analysis().resolve_callable(position)? {
        let parameters: Vec<ParameterInformation> = descriptor
            .params
            .iter()
            .map(|param| ParameterInformation {
                label: ParameterLabel::Simple(param.clone()),
                documentation: None,
            })
            .collect();

        let documentation = if let Some(doc) = descriptor.doc {
            Some(Documentation::MarkupContent(MarkupContent {
                kind: MarkupKind::Markdown,
                value: doc,
            }))
        } else {
            None
        };

        let sig_info = SignatureInformation {
            label: descriptor.label,
            documentation,
            parameters: Some(parameters),
        };

        Ok(Some(req::SignatureHelp {
            signatures: vec![sig_info],
            active_signature: Some(0),
            active_parameter: active_param.map(|a| a as u64),
        }))
    } else {
        Ok(None)
    }
}

pub fn handle_hover(
    world: ServerWorld,
    params: req::TextDocumentPositionParams,
) -> Result<Option<Hover>> {
    // TODO: Cut down on number of allocations
    let position = params.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(position.file_id);
    let rr = match world.analysis().approximately_resolve_symbol(position)? {
        None => return Ok(None),
        Some(it) => it,
    };
    let mut result = Vec::new();
    for (file_id, symbol) in rr.resolves_to {
        if let Some(docs) = world.analysis().doc_text_for(file_id, symbol)? {
            result.push(docs);
        }
    }
    let range = rr.reference_range.conv_with(&line_index);
    if result.len() > 0 {
        return Ok(Some(Hover {
            contents: HoverContents::Scalar(MarkedString::String(result.join("\n\n---\n"))),
            range: Some(range),
        }));
    }
    Ok(None)
}

/// Test doc comment
pub fn handle_prepare_rename(
    world: ServerWorld,
    params: req::TextDocumentPositionParams,
) -> Result<Option<PrepareRenameResponse>> {
    let position = params.try_conv_with(&world)?;

    // We support renaming references like handle_rename does.
    // In the future we may want to reject the renaming of things like keywords here too.
    let refs = world.analysis().find_all_refs(position)?;
    let r = match refs.first() {
        Some(r) => r,
        None => return Ok(None),
    };
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);
    let loc = to_location(r.0, r.1, &world, &line_index)?;

    Ok(Some(PrepareRenameResponse::Range(loc.range)))
}

pub fn handle_rename(world: ServerWorld, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);
    let offset = params.position.conv_with(&line_index);

    if params.new_name.is_empty() {
        return Err(LspError::new(
            ErrorCode::InvalidParams as i32,
            "New Name cannot be empty".into(),
        )
        .into());
    }

    let refs = world
        .analysis()
        .find_all_refs(FilePosition { file_id, offset })?;
    if refs.is_empty() {
        return Ok(None);
    }

    let mut changes = HashMap::new();
    for r in refs {
        if let Ok(loc) = to_location(r.0, r.1, &world, &line_index) {
            changes
                .entry(loc.uri)
                .or_insert_with(Vec::new)
                .push(TextEdit {
                    range: loc.range,
                    new_text: params.new_name.clone(),
                });
        }
    }

    Ok(Some(WorkspaceEdit {
        changes: Some(changes),

        // TODO: return this instead if client/server support it. See #144
        document_changes: None,
    }))
}

pub fn handle_references(
    world: ServerWorld,
    params: req::ReferenceParams,
) -> Result<Option<Vec<Location>>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);
    let offset = params.position.conv_with(&line_index);

    let refs = world
        .analysis()
        .find_all_refs(FilePosition { file_id, offset })?;

    Ok(Some(
        refs.into_iter()
            .filter_map(|r| to_location(r.0, r.1, &world, &line_index).ok())
            .collect(),
    ))
}

pub fn handle_code_action(
    world: ServerWorld,
    params: req::CodeActionParams,
) -> Result<Option<CodeActionResponse>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);
    let range = params.range.conv_with(&line_index);

    let assists = world.analysis().assists(file_id, range)?.into_iter();
    let fixes = world
        .analysis()
        .diagnostics(file_id)?
        .into_iter()
        .filter_map(|d| Some((d.range, d.fix?)))
        .filter(|(diag_range, _fix)| intersect(*diag_range, range).is_some())
        .map(|(_range, fix)| fix);

    let mut res = Vec::new();
    for source_edit in assists.chain(fixes) {
        let title = source_edit.label.clone();
        let edit = source_edit.try_conv_with(&world)?;
        let cmd = Command {
            title,
            command: "ra-lsp.applySourceChange".to_string(),
            arguments: Some(vec![to_value(edit).unwrap()]),
        };
        res.push(cmd);
    }

    Ok(Some(CodeActionResponse::Commands(res)))
}

pub fn publish_diagnostics(
    world: &ServerWorld,
    file_id: FileId,
) -> Result<req::PublishDiagnosticsParams> {
    let uri = world.file_id_to_uri(file_id)?;
    let line_index = world.analysis().file_line_index(file_id);
    let diagnostics = world
        .analysis()
        .diagnostics(file_id)?
        .into_iter()
        .map(|d| Diagnostic {
            range: d.range.conv_with(&line_index),
            severity: Some(to_diagnostic_severity(d.severity)),
            code: None,
            source: Some("rust-analyzer".to_string()),
            message: d.message,
            related_information: None,
        })
        .collect();
    Ok(req::PublishDiagnosticsParams { uri, diagnostics })
}

pub fn publish_decorations(
    world: &ServerWorld,
    file_id: FileId,
) -> Result<req::PublishDecorationsParams> {
    let uri = world.file_id_to_uri(file_id)?;
    Ok(req::PublishDecorationsParams {
        uri,
        decorations: highlight(&world, file_id)?,
    })
}

fn highlight(world: &ServerWorld, file_id: FileId) -> Result<Vec<Decoration>> {
    let line_index = world.analysis().file_line_index(file_id);
    let res = world
        .analysis()
        .highlight(file_id)?
        .into_iter()
        .map(|h| Decoration {
            range: h.range.conv_with(&line_index),
            tag: h.tag,
        })
        .collect();
    Ok(res)
}

fn to_diagnostic_severity(severity: Severity) -> DiagnosticSeverity {
    use ra_analysis::Severity::*;

    match severity {
        Error => DiagnosticSeverity::Error,
        Warning => DiagnosticSeverity::Warning,
        Information => DiagnosticSeverity::Information,
        Hint => DiagnosticSeverity::Hint,
    }
}
