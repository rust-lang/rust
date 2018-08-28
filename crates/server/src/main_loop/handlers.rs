use std::collections::HashMap;

use languageserver_types::{
    Diagnostic, DiagnosticSeverity, Url, DocumentSymbol,
    Command, TextDocumentIdentifier, WorkspaceEdit,
    SymbolInformation, Position, Location, TextEdit,
    CompletionItem,
};
use serde_json::{to_value, from_value};
use url_serde;
use libanalysis::{self, Query, FileId};
use libeditor;
use libsyntax2::{
    TextUnit,
    text_utils::contains_offset_nonstrict,
};

use ::{
    req::{self, Decoration}, Result,
    conv::{Conv, ConvWith, TryConvWith, MapConvWith, to_location},
    server_world::ServerWorld,
};

pub fn handle_syntax_tree(
    world: ServerWorld,
    params: req::SyntaxTreeParams,
) -> Result<String> {
    let id = params.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_syntax(id)?;
    Ok(libeditor::syntax_tree(&file))
}

pub fn handle_extend_selection(
    world: ServerWorld,
    params: req::ExtendSelectionParams,
) -> Result<req::ExtendSelectionResult> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_syntax(file_id)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let selections = params.selections.into_iter()
        .map_conv_with(&line_index)
        .map(|r| libeditor::extend_selection(&file, r).unwrap_or(r))
        .map_conv_with(&line_index)
        .collect();
    Ok(req::ExtendSelectionResult { selections })
}

pub fn handle_find_matching_brace(
    world: ServerWorld,
    params: req::FindMatchingBraceParams,
) -> Result<Vec<Position>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_syntax(file_id)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let res = params.offsets
        .into_iter()
        .map_conv_with(&line_index)
        .map(|offset| {
            libeditor::matching_brace(&file, offset).unwrap_or(offset)
        })
        .map_conv_with(&line_index)
        .collect();
    Ok(res)
}

pub fn handle_join_lines(
    world: ServerWorld,
    params: req::JoinLinesParams,
) -> Result<Vec<TextEdit>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_syntax(file_id)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let range = params.range.conv_with(&line_index);
    let res = libeditor::join_lines(&file, range);
    Ok(res.edit.conv_with(&line_index))
}

pub fn handle_document_symbol(
    world: ServerWorld,
    params: req::DocumentSymbolParams,
) -> Result<Option<req::DocumentSymbolResponse>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_syntax(file_id)?;
    let line_index = world.analysis().file_line_index(file_id)?;

    let mut parents: Vec<(DocumentSymbol, Option<usize>)> = Vec::new();

    for symbol in libeditor::file_structure(&file) {
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

pub fn handle_code_action(
    world: ServerWorld,
    params: req::CodeActionParams,
) -> Result<Option<Vec<Command>>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_syntax(file_id)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let offset = params.range.conv_with(&line_index).start();
    let mut res = Vec::new();

    let actions = &[
        (ActionId::FlipComma, libeditor::flip_comma(&file, offset).is_some()),
        (ActionId::AddDerive, libeditor::add_derive(&file, offset).is_some()),
        (ActionId::AddImpl, libeditor::add_impl(&file, offset).is_some()),
    ];

    for (id, edit) in actions {
        if *edit {
            let cmd = apply_code_action_cmd(*id, params.text_document.clone(), offset);
            res.push(cmd);
        }
    }

    for (diag, quick_fix) in world.analysis().diagnostics(file_id)? {
        let quick_fix = match quick_fix {
            Some(quick_fix) => quick_fix,
            None => continue,
        };
        if !contains_offset_nonstrict(diag.range, offset) {
            continue;
        }
        let mut ops = Vec::new();
        for op in quick_fix.fs_ops {
            let op = match op {
                libanalysis::FsOp::CreateFile { anchor, path } => {
                    let uri = world.file_id_to_uri(anchor)?;
                    let path = &path.as_str()[3..]; // strip `../` b/c url is weird
                    let uri = uri.join(path)?;
                    FsOp::CreateFile { uri }
                },
                libanalysis::FsOp::MoveFile { file, path } => {
                    let src = world.file_id_to_uri(file)?;
                    let path = &path.as_str()[3..]; // strip `../` b/c url is weird
                    let dst = src.join(path)?;
                    FsOp::MoveFile { src, dst }
                },
            };
            ops.push(op)
        }
        let cmd = Command {
            title: "Create module".to_string(),
            command: "libsyntax-rust.fsEdit".to_string(),
            arguments: Some(vec![to_value(ops).unwrap()]),
        };
        res.push(cmd)
    }
    return Ok(Some(res));
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum FsOp {
    CreateFile {
        #[serde(with = "url_serde")]
        uri: Url
    },
    MoveFile {
        #[serde(with = "url_serde")]
        src: Url,
        #[serde(with = "url_serde")]
        dst: Url,
    }
}

pub fn handle_runnables(
    world: ServerWorld,
    params: req::RunnablesParams,
) -> Result<Vec<req::Runnable>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_syntax(file_id)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let offset = params.position.map(|it| it.conv_with(&line_index));
    let mut res = Vec::new();
    for runnable in libeditor::runnables(&file) {
        if let Some(offset) = offset {
            if !contains_offset_nonstrict(runnable.range, offset) {
                continue;
            }
        }

        let r = req::Runnable {
            range: runnable.range.conv_with(&line_index),
            label: match &runnable.kind {
                libeditor::RunnableKind::Test { name } =>
                    format!("test {}", name),
                libeditor::RunnableKind::Bin =>
                    "run binary".to_string(),
            },
            bin: "cargo".to_string(),
            args: match runnable.kind {
                libeditor::RunnableKind::Test { name } => {
                    vec![
                        "test".to_string(),
                        "--".to_string(),
                        name,
                        "--nocapture".to_string(),
                    ]
                }
                libeditor::RunnableKind::Bin => vec!["run".to_string()]
            },
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
}

pub fn handle_workspace_symbol(
    world: ServerWorld,
    params: req::WorkspaceSymbolParams,
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
    let mut res = exec_query(&world, query)?;
    if res.is_empty() && !all_symbols {
        let mut query = Query::new(params.query);
        query.limit(128);
        res = exec_query(&world, query)?;
    }

    return Ok(Some(res));

    fn exec_query(world: &ServerWorld, query: Query) -> Result<Vec<SymbolInformation>> {
        let mut res = Vec::new();
        for (file_id, symbol) in world.analysis().world_symbols(query) {
            let line_index = world.analysis().file_line_index(file_id)?;
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
) -> Result<Option<req::GotoDefinitionResponse>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let offset = params.position.conv_with(&line_index);
    let mut res = Vec::new();
    for (file_id, symbol) in world.analysis().approximately_resolve_symbol(file_id, offset)? {
        let line_index = world.analysis().file_line_index(file_id)?;
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
) -> Result<Vec<Location>> {
    let file_id = params.try_conv_with(&world)?;
    let mut res = Vec::new();
    for (file_id, symbol) in world.analysis().parent_module(file_id) {
        let line_index = world.analysis().file_line_index(file_id)?;
        let location = to_location(
            file_id, symbol.node_range,
            &world, &line_index
        )?;
        res.push(location);
    }
    Ok(res)
}

pub fn handle_completion(
    world: ServerWorld,
    params: req::CompletionParams,
) -> Result<Option<req::CompletionResponse>> {
    let file_id = params.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_syntax(file_id)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let offset = params.position.conv_with(&line_index);
    let items = match libeditor::scope_completion(&file, offset) {
        None => return Ok(None),
        Some(items) => items,
    };
    let items = items.into_iter()
        .map(|item| CompletionItem {
            label: item.name,
            .. Default::default()
        })
        .collect();

    Ok(Some(req::CompletionResponse::Array(items)))
}

pub fn handle_on_type_formatting(
    world: ServerWorld,
    params: req::DocumentOnTypeFormattingParams,
) -> Result<Option<Vec<TextEdit>>> {
    if params.ch != "=" {
        return Ok(None);
    }

    let file_id = params.text_document.try_conv_with(&world)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let offset = params.position.conv_with(&line_index);
    let file = world.analysis().file_syntax(file_id)?;
    let action = match libeditor::on_eq_typed(&file, offset) {
        None => return Ok(None),
        Some(action) => action,
    };
    Ok(Some(action.edit.conv_with(&line_index)))
}

pub fn handle_execute_command(
    world: ServerWorld,
    mut params: req::ExecuteCommandParams,
) -> Result<(req::ApplyWorkspaceEditParams, Option<Position>)> {
    if params.command.as_str() != "apply_code_action" {
        bail!("unknown cmd: {:?}", params.command);
    }
    if params.arguments.len() != 1 {
        bail!("expected single arg, got {}", params.arguments.len());
    }
    let arg = params.arguments.pop().unwrap();
    let arg: ActionRequest = from_value(arg)?;
    let file_id = arg.text_document.try_conv_with(&world)?;
    let file = world.analysis().file_syntax(file_id)?;
    let action_result = match arg.id {
        ActionId::FlipComma => libeditor::flip_comma(&file, arg.offset).map(|f| f()),
        ActionId::AddDerive => libeditor::add_derive(&file, arg.offset).map(|f| f()),
        ActionId::AddImpl => libeditor::add_impl(&file, arg.offset).map(|f| f()),
    }.ok_or_else(|| format_err!("command not applicable"))?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let mut changes = HashMap::new();
    changes.insert(
        arg.text_document.uri,
        action_result.edit.conv_with(&line_index),
    );
    let edit = WorkspaceEdit {
        changes: Some(changes),
        document_changes: None,
    };
    let edit = req::ApplyWorkspaceEditParams { edit };
    let cursor_pos = action_result.cursor_position
        .map(|off| off.conv_with(&line_index));
    Ok((edit, cursor_pos))
}

#[derive(Serialize, Deserialize)]
struct ActionRequest {
    id: ActionId,
    text_document: TextDocumentIdentifier,
    offset: TextUnit,
}

fn apply_code_action_cmd(id: ActionId, doc: TextDocumentIdentifier, offset: TextUnit) -> Command {
    let action_request = ActionRequest { id, text_document: doc, offset };
    Command {
        title: id.title().to_string(),
        command: "apply_code_action".to_string(),
        arguments: Some(vec![to_value(action_request).unwrap()]),
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
enum ActionId {
    FlipComma,
    AddDerive,
    AddImpl,
}

impl ActionId {
    fn title(&self) -> &'static str {
        match *self {
            ActionId::FlipComma => "Flip `,`",
            ActionId::AddDerive => "Add `#[derive]`",
            ActionId::AddImpl => "Add impl",
        }
    }
}

pub fn publish_diagnostics(
    world: ServerWorld,
    uri: Url
) -> Result<req::PublishDiagnosticsParams> {
    let file_id = world.uri_to_file_id(&uri)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let diagnostics = world.analysis().diagnostics(file_id)?
        .into_iter()
        .map(|(d, _quick_fix)| Diagnostic {
            range: d.range.conv_with(&line_index),
            severity: Some(DiagnosticSeverity::Error),
            code: None,
            source: Some("libsyntax2".to_string()),
            message: d.msg,
            related_information: None,
        }).collect();
    Ok(req::PublishDiagnosticsParams { uri, diagnostics })
}

pub fn handle_decorations(
    world: ServerWorld,
    params: TextDocumentIdentifier,
) -> Result<Vec<Decoration>> {
    let file_id = params.try_conv_with(&world)?;
    highlight(&world, file_id)
}

pub fn publish_decorations(
    world: ServerWorld,
    uri: Url
) -> Result<req::PublishDecorationsParams> {
    let file_id = world.uri_to_file_id(&uri)?;
    Ok(req::PublishDecorationsParams {
        uri,
        decorations: highlight(&world, file_id)?
    })
}

fn highlight(world: &ServerWorld, file_id: FileId) -> Result<Vec<Decoration>> {
    let file = world.analysis().file_syntax(file_id)?;
    let line_index = world.analysis().file_line_index(file_id)?;
    let res = libeditor::highlight(&file)
        .into_iter()
        .map(|h| Decoration {
            range: h.range.conv_with(&line_index),
            tag: h.tag,
        }).collect();
    Ok(res)
}
