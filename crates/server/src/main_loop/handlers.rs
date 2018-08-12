use languageserver_types::{
    Diagnostic, DiagnosticSeverity, Url, DocumentSymbol,
    Command
};
use libanalysis::World;
use libeditor;
use serde_json::to_value;

use ::{
    req::{self, Decoration}, Result,
    util::FilePath,
    conv::{Conv, ConvWith},
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
        .map(|r| r.conv_with(&line_index))
        .map(|r| libeditor::extend_selection(&file, r).unwrap_or(r))
        .map(|r| r.conv_with(&line_index))
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
        let doc_symbol = DocumentSymbol {
            name: symbol.name.clone(),
            detail: Some(symbol.name),
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
        Some(vec![apply_code_action_cmd(ActionId::FlipComma)])
    } else {
        None
    };
    Ok(ret)
}

fn apply_code_action_cmd(id: ActionId) -> Command {
    Command {
        title: id.title().to_string(),
        command: "apply_code_action".to_string(),
        arguments: Some(vec![to_value(id).unwrap()]),
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
