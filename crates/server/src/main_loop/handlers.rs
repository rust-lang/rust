use std::collections::HashMap;

use languageserver_types::{
    Diagnostic, DiagnosticSeverity, DocumentSymbol,
    Command, TextDocumentIdentifier,
    SymbolInformation, Position, Location, TextEdit,
    CompletionItem, InsertTextFormat, CompletionItemKind,
};
use serde_json::to_value;
use libanalysis::{Query, FileId, RunnableKind, JobToken};
use libsyntax2::{
    text_utils::contains_offset_nonstrict,
};

use ::{
    req::{self, Decoration}, Result,
    conv::{Conv, ConvWith, TryConvWith, MapConvWith, to_location},
    server_world::ServerWorld,
    project_model::TargetKind,
};

pub fn handle_syntax_tree(
    world: ServerWorld,
    params: req::SyntaxTreeParams,
    _token: JobToken,
) -> Result<String> {
    let id = params.text_document.try_conv_with(&world)?;
    let res = world.analysis().syntax_tree(id);
    Ok(res)
}

pub fn handle_extend_selection(
    world: ServerWorld,
    params: req::ExtendSelectionParams,
    _token: JobToken,
) -> Result<req::ExtendSelectionResult> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_syntax(file_id);
    let line_index = world.analysis().file_line_index(file_id);
    let selections = params.selections.into_iter()
        .map_conv_with(&line_index)
        .map(|r| world.analysis().extend_selection(&file, r))
        .map_conv_with(&line_index)
        .collect();
    Ok(req::ExtendSelectionResult { selections })
}

pub fn handle_find_matching_brace(
    world: ServerWorld,
    params: req::FindMatchingBraceParams,
    _token: JobToken,
) -> Result<Vec<Position>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_syntax(file_id);
    let line_index = world.analysis().file_line_index(file_id);
    let res = params.offsets
        .into_iter()
        .map_conv_with(&line_index)
        .map(|offset| {
            world.analysis().matching_brace(&file, offset).unwrap_or(offset)
        })
        .map_conv_with(&line_index)
        .collect();
    Ok(res)
}

pub fn handle_join_lines(
    world: ServerWorld,
    params: req::JoinLinesParams,
    _token: JobToken,
) -> Result<req::SourceChange> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);
    let range = params.range.conv_with(&line_index);
    world.analysis().join_lines(file_id, range)
        .try_conv_with(&world)
}

pub fn handle_on_type_formatting(
    world: ServerWorld,
    params: req::DocumentOnTypeFormattingParams,
    _token: JobToken,
) -> Result<Option<Vec<TextEdit>>> {
    if params.ch != "=" {
        return Ok(None);
    }

    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);
    let offset = params.position.conv_with(&line_index);
    let edits = match world.analysis().on_eq_typed(file_id, offset) {
        None => return Ok(None),
        Some(mut action) => action.source_file_edits.pop().unwrap().edits,
    };
    let edits = edits.into_iter().map_conv_with(&line_index).collect();
    Ok(Some(edits))
}

pub fn handle_document_symbol(
    world: ServerWorld,
    params: req::DocumentSymbolParams,
    _token: JobToken,
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
    token: JobToken,
) -> Result<Option<Vec<SymbolInformation>>> {
    let all_symbols = params.query.contains("#");
    let query = {
        let query: String = params.query.chars()
            .filter(|&c| c != '#')
            .collect();
        let mut q = Query::new(query);
        if !all_symbols {
            q.only_types();
        }
        q.limit(128);
        q
    };
    let mut res = exec_query(&world, query, &token)?;
    if res.is_empty() && !all_symbols {
        let mut query = Query::new(params.query);
        query.limit(128);
        res = exec_query(&world, query, &token)?;
    }

    return Ok(Some(res));

    fn exec_query(world: &ServerWorld, query: Query, token: &JobToken) -> Result<Vec<SymbolInformation>> {
        let mut res = Vec::new();
        for (file_id, symbol) in world.analysis().symbol_search(query, token) {
            let line_index = world.analysis().file_line_index(file_id);
            let info = SymbolInformation {
                name: symbol.name.to_string(),
                kind: symbol.kind.conv(),
                location: to_location(
                    file_id, symbol.node_range,
                    world, &line_index
                )?,
                container_name: None,
            };
            res.push(info);
        };
        Ok(res)
    }
}

pub fn handle_goto_definition(
    world: ServerWorld,
    params: req::TextDocumentPositionParams,
    token: JobToken,
) -> Result<Option<req::GotoDefinitionResponse>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);
    let offset = params.position.conv_with(&line_index);
    let mut res = Vec::new();
    for (file_id, symbol) in world.analysis().approximately_resolve_symbol(file_id, offset, &token) {
        let line_index = world.analysis().file_line_index(file_id);
        let location = to_location(
            file_id, symbol.node_range,
            &world, &line_index,
        )?;
        res.push(location)
    }
    Ok(Some(req::GotoDefinitionResponse::Array(res)))
}

pub fn handle_parent_module(
    world: ServerWorld,
    params: TextDocumentIdentifier,
    _token: JobToken,
) -> Result<Vec<Location>> {
    let file_id = params.try_conv_with(&world)?;
    let mut res = Vec::new();
    for (file_id, symbol) in world.analysis().parent_module(file_id) {
        let line_index = world.analysis().file_line_index(file_id);
        let location = to_location(
            file_id, symbol.node_range,
            &world, &line_index
        )?;
        res.push(location);
    }
    Ok(res)
}

pub fn handle_runnables(
    world: ServerWorld,
    params: req::RunnablesParams,
    _token: JobToken,
) -> Result<Vec<req::Runnable>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);
    let offset = params.position.map(|it| it.conv_with(&line_index));
    let mut res = Vec::new();
    for runnable in world.analysis().runnables(file_id) {
        if let Some(offset) = offset {
            if !contains_offset_nonstrict(runnable.range, offset) {
                continue;
            }
        }

        let args = runnable_args(&world, file_id, &runnable.kind);

        let r = req::Runnable {
            range: runnable.range.conv_with(&line_index),
            label: match &runnable.kind {
                RunnableKind::Test { name } =>
                    format!("test {}", name),
                RunnableKind::Bin =>
                    "run binary".to_string(),
            },
            bin: "cargo".to_string(),
            args,
            env: {
                let mut m = HashMap::new();
                m.insert(
                    "RUST_BACKTRACE".to_string(),
                    "short".to_string(),
                );
                m
            }
        };
        res.push(r);
    }
    return Ok(res);

    fn runnable_args(world: &ServerWorld, file_id: FileId, kind: &RunnableKind) -> Vec<String> {
        let spec = if let Some(&crate_id) = world.analysis().crate_for(file_id).first() {
            let file_id = world.analysis().crate_root(crate_id);
            let path = world.path_map.get_path(file_id);
            world.workspaces.iter()
                .filter_map(|ws| {
                    let tgt = ws.target_by_root(path)?;
                    Some((tgt.package(ws).name(ws).clone(), tgt.name(ws).clone(), tgt.kind(ws)))
                })
                .next()
        } else {
            None
        };
        let mut res = Vec::new();
        match kind {
                RunnableKind::Test { name } => {
                    res.push("test".to_string());
                    if let Some((pkg_name, tgt_name, tgt_kind)) = spec {
                        spec_args(pkg_name, tgt_name, tgt_kind, &mut res);
                    }
                    res.push("--".to_string());
                    res.push(name.to_string());
                    res.push("--nocapture".to_string());
                }
                RunnableKind::Bin => {
                    res.push("run".to_string());
                    if let Some((pkg_name, tgt_name, tgt_kind)) = spec {
                        spec_args(pkg_name, tgt_name, tgt_kind, &mut res);
                    }
                }
            }
        res
    }

    fn spec_args(pkg_name: &str, tgt_name: &str, tgt_kind: TargetKind, buf: &mut Vec<String>) {
        buf.push("--package".to_string());
        buf.push(pkg_name.to_string());
        match tgt_kind {
            TargetKind::Bin => {
                buf.push("--bin".to_string());
                buf.push(tgt_name.to_string());
            }
            TargetKind::Test => {
                buf.push("--test".to_string());
                buf.push(tgt_name.to_string());
            }
            TargetKind::Bench => {
                buf.push("--bench".to_string());
                buf.push(tgt_name.to_string());
            }
            TargetKind::Example => {
                buf.push("--example".to_string());
                buf.push(tgt_name.to_string());
            }
            TargetKind::Lib => {
                buf.push("--lib".to_string());
            }
            TargetKind::Other => (),
        }
    }
}

pub fn handle_decorations(
    world: ServerWorld,
    params: TextDocumentIdentifier,
    _token: JobToken,
) -> Result<Vec<Decoration>> {
    let file_id = params.try_conv_with(&world)?;
    Ok(highlight(&world, file_id))
}

pub fn handle_completion(
    world: ServerWorld,
    params: req::CompletionParams,
    _token: JobToken,
) -> Result<Option<req::CompletionResponse>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);
    let offset = params.position.conv_with(&line_index);
    let items = match world.analysis().completions(file_id, offset) {
        None => return Ok(None),
        Some(items) => items,
    };
    let items = items.into_iter()
        .map(|item| {
            let mut res = CompletionItem {
                label: item.label,
                filter_text: item.lookup,
                .. Default::default()
            };
            if let Some(snip) = item.snippet {
                res.insert_text = Some(snip);
                res.insert_text_format = Some(InsertTextFormat::Snippet);
                res.kind = Some(CompletionItemKind::Keyword);
            };
            res
        })
        .collect();

    Ok(Some(req::CompletionResponse::Array(items)))
}

pub fn handle_code_action(
    world: ServerWorld,
    params: req::CodeActionParams,
    _token: JobToken,
) -> Result<Option<Vec<Command>>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id);
    let offset = params.range.conv_with(&line_index).start();

    let assists = world.analysis().assists(file_id, offset).into_iter();
    let fixes = world.analysis().diagnostics(file_id).into_iter()
        .filter_map(|d| Some((d.range, d.fix?)))
        .filter(|(range, _fix)| contains_offset_nonstrict(*range, offset))
        .map(|(_range, fix)| fix);

    let mut res = Vec::new();
    for source_edit in assists.chain(fixes) {
        let title = source_edit.label.clone();
        let edit = source_edit.try_conv_with(&world)?;
        let cmd = Command {
            title,
            command: "libsyntax-rust.applySourceChange".to_string(),
            arguments: Some(vec![to_value(edit).unwrap()]),
        };
        res.push(cmd);
    }

    Ok(Some(res))
}

pub fn publish_diagnostics(
    world: ServerWorld,
    file_id: FileId,
) -> Result<req::PublishDiagnosticsParams> {
    let uri = world.file_id_to_uri(file_id)?;
    let line_index = world.analysis().file_line_index(file_id);
    let diagnostics = world.analysis().diagnostics(file_id)
        .into_iter()
        .map(|d| Diagnostic {
            range: d.range.conv_with(&line_index),
            severity: Some(DiagnosticSeverity::Error),
            code: None,
            source: Some("libsyntax2".to_string()),
            message: d.message,
            related_information: None,
        }).collect();
    Ok(req::PublishDiagnosticsParams { uri, diagnostics })
}

pub fn publish_decorations(
    world: ServerWorld,
    file_id: FileId,
) -> Result<req::PublishDecorationsParams> {
    let uri = world.file_id_to_uri(file_id)?;
    Ok(req::PublishDecorationsParams {
        uri,
        decorations: highlight(&world, file_id),
    })
}

fn highlight(world: &ServerWorld, file_id: FileId) -> Vec<Decoration> {
    let line_index = world.analysis().file_line_index(file_id);
    world.analysis().highlight(file_id)
        .into_iter()
        .map(|h| Decoration {
            range: h.range.conv_with(&line_index),
            tag: h.tag,
        }).collect()
}
