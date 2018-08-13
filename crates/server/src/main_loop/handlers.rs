use std::collections::HashMap;

use languageserver_types::{
    Diagnostic, DiagnosticSeverity, Url, DocumentSymbol,
    Command, TextDocumentIdentifier, WorkspaceEdit,
    SymbolInformation,
};
use libanalysis::{World, Query};
use libeditor;
use libsyntax2::TextUnit;
use serde_json::{to_value, from_value};

use ::{
    req::{self, Decoration}, Result,
    util::FilePath,
    conv::{Conv, ConvWith, TryConvWith, MapConvWith},
};

pub fn handle_syntax_tree(
    world: World,
    params: req::SyntaxTreeParams,
) -> Result<String> {
    let path = params.text_document.file_path()?;
    let file = world.file_syntax(&path)?;
    Ok(libeditor::syntax_tree(&file))
}

pub fn handle_extend_selection(
    world: World,
    params: req::ExtendSelectionParams,
) -> Result<req::ExtendSelectionResult> {
    let path = params.text_document.file_path()?;
    let file = world.file_syntax(&path)?;
    let line_index = world.file_line_index(&path)?;
    let selections = params.selections.into_iter()
        .map_conv_with(&line_index)
        .map(|r| libeditor::extend_selection(&file, r).unwrap_or(r))
        .map_conv_with(&line_index)
        .collect();
    Ok(req::ExtendSelectionResult { selections })
}

pub fn handle_document_symbol(
    world: World,
    params: req::DocumentSymbolParams,
) -> Result<Option<req::DocumentSymbolResponse>> {
    let path = params.text_document.file_path()?;
    let file = world.file_syntax(&path)?;
    let line_index = world.file_line_index(&path)?;

    let mut res: Vec<DocumentSymbol> = Vec::new();

    for symbol in libeditor::file_symbols(&file) {
        let name = symbol.name.to_string();
        let doc_symbol = DocumentSymbol {
            name: name.clone(),
            detail: Some(name),
            kind: symbol.kind.conv(),
            deprecated: None,
            range: symbol.node_range.conv_with(&line_index),
            selection_range: symbol.name_range.conv_with(&line_index),
            children: None,
        };
        if let Some(idx) = symbol.parent {
            let children = &mut res[idx].children;
            if children.is_none() {
                *children = Some(Vec::new());
            }
            children.as_mut().unwrap().push(doc_symbol);
        } else {
            res.push(doc_symbol);
        }
    }
    Ok(Some(req::DocumentSymbolResponse::Nested(res)))
}

pub fn handle_code_action(
    world: World,
    params: req::CodeActionParams,
) -> Result<Option<Vec<Command>>> {
    let path = params.text_document.file_path()?;
    let file = world.file_syntax(&path)?;
    let line_index = world.file_line_index(&path)?;
    let offset = params.range.conv_with(&line_index).start();
    let ret = if libeditor::flip_comma(&file, offset).is_some() {
        let cmd = apply_code_action_cmd(
            ActionId::FlipComma,
            params.text_document,
            offset,
        );
        Some(vec![cmd])
    } else {
        None
    };
    Ok(ret)
}

pub fn handle_workspace_symbol(
    world: World,
    params: req::WorkspaceSymbolParams,
) -> Result<Option<Vec<SymbolInformation>>> {
    let mut acc = Vec::new();

    let query = {
        let all_symbols = params.query.contains("#");
        let query: String = params.query.chars()
            .filter(|&c| c != '#')
            .collect();
        let mut q = Query::new(query);
        if !all_symbols {
            q.only_types();
        }
        q
    };

    for (path, symbol) in world.world_symbols(query).take(128) {
        let line_index = world.file_line_index(path)?;
        let info = SymbolInformation {
            name: symbol.name.to_string(),
            kind: symbol.kind.conv(),
            location: (path, symbol.node_range).try_conv_with(&line_index)?,
            container_name: None,
        };
        acc.push(info);
    };

    Ok(Some(acc))
}

pub fn handle_goto_definition(
    world: World,
    params: req::TextDocumentPositionParams,
) -> Result<Option<req::GotoDefinitionResponse>> {
    let path = params.text_document.file_path()?;
    let line_index = world.file_line_index(&path)?;
    let offset = params.position.conv_with(&line_index);
    let mut res = Vec::new();
    for (path, symbol) in world.approximately_resolve_symbol(&path, offset)? {
        let line_index = world.file_line_index(path)?;
        let location = (path, symbol.node_range).try_conv_with(&line_index)?;
        res.push(location)
    }
    Ok(Some(req::GotoDefinitionResponse::Array(res)))
}

pub fn handle_execute_command(
    world: World,
    mut params: req::ExecuteCommandParams,
) -> Result<req::ApplyWorkspaceEditParams> {
    if params.command.as_str() != "apply_code_action" {
        bail!("unknown cmd: {:?}", params.command);
    }
    if params.arguments.len() != 1 {
        bail!("expected single arg, got {}", params.arguments.len());
    }
    let arg = params.arguments.pop().unwrap();
    let arg: ActionRequest = from_value(arg)?;
    match arg.id {
        ActionId::FlipComma => {
            let path = arg.text_document.file_path()?;
            let file = world.file_syntax(&path)?;
            let line_index = world.file_line_index(&path)?;
            let edit = match libeditor::flip_comma(&file, arg.offset) {
                Some(edit) => edit(),
                None => bail!("command not applicable"),
            };
            let mut changes = HashMap::new();
            changes.insert(
                arg.text_document.uri,
                edit.conv_with(&line_index),
            );
            let edit = WorkspaceEdit {
                changes: Some(changes),
                document_changes: None,
            };

            Ok(req::ApplyWorkspaceEditParams { edit })
        }
    }
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
    FlipComma
}

impl ActionId {
    fn title(&self) -> &'static str {
        match *self {
            ActionId::FlipComma => "Flip `,`",
        }
    }
}

pub fn publish_diagnostics(world: World, uri: Url) -> Result<req::PublishDiagnosticsParams> {
    let path = uri.file_path()?;
    let file = world.file_syntax(&path)?;
    let line_index = world.file_line_index(&path)?;
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

pub fn publish_decorations(world: World, uri: Url) -> Result<req::PublishDecorationsParams> {
    let path = uri.file_path()?;
    let file = world.file_syntax(&path)?;
    let line_index = world.file_line_index(&path)?;
    let decorations = libeditor::highlight(&file)
        .into_iter()
        .map(|h| Decoration {
            range: h.range.conv_with(&line_index),
            tag: h.tag,
        }).collect();
    Ok(req::PublishDecorationsParams { uri, decorations })
}
