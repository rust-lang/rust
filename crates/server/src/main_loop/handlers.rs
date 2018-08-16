use std::collections::HashMap;

use languageserver_types::{
    Diagnostic, DiagnosticSeverity, Url, DocumentSymbol,
    Command, TextDocumentIdentifier, WorkspaceEdit,
    SymbolInformation, Position,
};
use libanalysis::{World, Query};
use libeditor::{self, CursorPosition};
use libsyntax2::TextUnit;
use serde_json::{to_value, from_value};

use ::{
    PathMap,
    req::{self, Decoration}, Result,
    conv::{Conv, ConvWith, TryConvWith, MapConvWith, to_location},
};

pub fn handle_syntax_tree(
    world: World,
    path_map: PathMap,
    params: req::SyntaxTreeParams,
) -> Result<String> {
    let id = params.text_document.try_conv_with(&path_map)?;
    let file = world.file_syntax(id)?;
    Ok(libeditor::syntax_tree(&file))
}

pub fn handle_extend_selection(
    world: World,
    path_map: PathMap,
    params: req::ExtendSelectionParams,
) -> Result<req::ExtendSelectionResult> {
    let file_id = params.text_document.try_conv_with(&path_map)?;
    let file = world.file_syntax(file_id)?;
    let line_index = world.file_line_index(file_id)?;
    let selections = params.selections.into_iter()
        .map_conv_with(&line_index)
        .map(|r| libeditor::extend_selection(&file, r).unwrap_or(r))
        .map_conv_with(&line_index)
        .collect();
    Ok(req::ExtendSelectionResult { selections })
}

pub fn handle_find_matching_brace(
    world: World,
    path_map: PathMap,
    params: req::FindMatchingBraceParams,
) -> Result<Vec<Position>> {
    let file_id = params.text_document.try_conv_with(&path_map)?;
    let file = world.file_syntax(file_id)?;
    let line_index = world.file_line_index(file_id)?;
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

pub fn handle_document_symbol(
    world: World,
    path_map: PathMap,
    params: req::DocumentSymbolParams,
) -> Result<Option<req::DocumentSymbolResponse>> {
    let file_id = params.text_document.try_conv_with(&path_map)?;
    let file = world.file_syntax(file_id)?;
    let line_index = world.file_line_index(file_id)?;

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
    world: World,
    path_map: PathMap,
    params: req::CodeActionParams,
) -> Result<Option<Vec<Command>>> {
    let file_id = params.text_document.try_conv_with(&path_map)?;
    let file = world.file_syntax(file_id)?;
    let line_index = world.file_line_index(file_id)?;
    let offset = params.range.conv_with(&line_index).start();
    let mut ret = Vec::new();

    let actions = &[
        (ActionId::FlipComma, libeditor::flip_comma(&file, offset).is_some()),
        (ActionId::AddDerive, libeditor::add_derive(&file, offset).is_some()),
    ];

    for (id, edit) in actions {
        if *edit {
            let cmd = apply_code_action_cmd(*id, params.text_document.clone(), offset);
            ret.push(cmd);
        }
    }
    return Ok(Some(ret));
}

pub fn handle_workspace_symbol(
    world: World,
    path_map: PathMap,
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
    let mut res = exec_query(&world, &path_map, query)?;
    if res.is_empty() && !all_symbols {
        let mut query = Query::new(params.query);
        query.limit(128);
        res = exec_query(&world, &path_map, query)?;
    }

    return Ok(Some(res));

    fn exec_query(world: &World, path_map: &PathMap, query: Query) -> Result<Vec<SymbolInformation>> {
        let mut res = Vec::new();
        for (file_id, symbol) in world.world_symbols(query) {
            let line_index = world.file_line_index(file_id)?;
            let info = SymbolInformation {
                name: symbol.name.to_string(),
                kind: symbol.kind.conv(),
                location: to_location(
                    file_id, symbol.node_range,
                    path_map, &line_index
                )?,
                container_name: None,
            };
            res.push(info);
        };
        Ok(res)
    }
}

pub fn handle_goto_definition(
    world: World,
    path_map: PathMap,
    params: req::TextDocumentPositionParams,
) -> Result<Option<req::GotoDefinitionResponse>> {
    let file_id = params.text_document.try_conv_with(&path_map)?;
    let line_index = world.file_line_index(file_id)?;
    let offset = params.position.conv_with(&line_index);
    let mut res = Vec::new();
    for (file_id, symbol) in world.approximately_resolve_symbol(file_id, offset)? {
        let line_index = world.file_line_index(file_id)?;
        let location = to_location(
            file_id, symbol.node_range,
            &path_map, &line_index,
        )?;
        res.push(location)
    }
    Ok(Some(req::GotoDefinitionResponse::Array(res)))
}

pub fn handle_execute_command(
    world: World,
    path_map: PathMap,
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
    let file_id = arg.text_document.try_conv_with(&path_map)?;
    let file = world.file_syntax(file_id)?;
    let action_result = match arg.id {
        ActionId::FlipComma => libeditor::flip_comma(&file, arg.offset).map(|f| f()),
        ActionId::AddDerive => libeditor::add_derive(&file, arg.offset).map(|f| f()),
    }.ok_or_else(|| format_err!("command not applicable"))?;
    let line_index = world.file_line_index(file_id)?;
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
    let cursor_pos = match action_result.cursor_position {
        CursorPosition::Same => None,
        CursorPosition::Offset(offset) => Some(offset.conv_with(&line_index)),
    };

    Ok((edit, cursor_pos))
}

#[derive(Serialize, Deserialize)]
struct ActionRequest {
    id: ActionId,
    text_document: TextDocumentIdentifier,
    offset: TextUnit,
}

fn apply_code_action_cmd(id: ActionId, doc: TextDocumentIdentifier, offset: TextUnit) -> Command {
    let action_request = ActionRequest {
        id,
        text_document: doc,
        offset,
    };
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
}

impl ActionId {
    fn title(&self) -> &'static str {
        match *self {
            ActionId::FlipComma => "Flip `,`",
            ActionId::AddDerive => "Add `#[derive]`",
        }
    }
}

pub fn publish_diagnostics(
    world: World,
    path_map: PathMap,
    uri: Url
) -> Result<req::PublishDiagnosticsParams> {
    let file_id = uri.try_conv_with(&path_map)?;
    let file = world.file_syntax(file_id)?;
    let line_index = world.file_line_index(file_id)?;
    let diagnostics = libeditor::diagnostics(&file)
        .into_iter()
        .map(|d| Diagnostic {
            range: d.range.conv_with(&line_index),
            severity: Some(DiagnosticSeverity::Error),
            code: None,
            source: Some("libsyntax2".to_string()),
            message: d.msg,
            related_information: None,
        }).collect();
    Ok(req::PublishDiagnosticsParams { uri, diagnostics })
}

pub fn publish_decorations(
    world: World,
    path_map: PathMap,
    uri: Url
) -> Result<req::PublishDecorationsParams> {
    let file_id = uri.try_conv_with(&path_map)?;
    let file = world.file_syntax(file_id)?;
    let line_index = world.file_line_index(file_id)?;
    let decorations = libeditor::highlight(&file)
        .into_iter()
        .map(|h| Decoration {
            range: h.range.conv_with(&line_index),
            tag: h.tag,
        }).collect();
    Ok(req::PublishDecorationsParams { uri, decorations })
}
