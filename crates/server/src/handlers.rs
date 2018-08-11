use languageserver_types::{Range, Position, Diagnostic, DiagnosticSeverity, Url, DocumentSymbol, SymbolKind};
use libsyntax2::SyntaxKind;
use libanalysis::World;
use libeditor::{self, LineIndex, LineCol, TextRange, TextUnit};

use ::{
    req::{self, Decoration}, Result,
    util::FilePath,
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
        .map(|r| {
            let r = to_text_range(&line_index, r);
            let r = libeditor::extend_selection(&file, r).unwrap_or(r);
            to_vs_range(&line_index, r)
        })
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
            kind: to_symbol_kind(symbol.kind),
            deprecated: None,
            range: to_vs_range(&line_index, symbol.node_range),
            selection_range: to_vs_range(&line_index, symbol.name_range),
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

fn to_symbol_kind(kind: SyntaxKind) -> SymbolKind {
    match kind {
        SyntaxKind::FUNCTION => SymbolKind::Function,
        SyntaxKind::STRUCT => SymbolKind::Struct,
        SyntaxKind::ENUM => SymbolKind::Enum,
        SyntaxKind::TRAIT => SymbolKind::Interface,
        SyntaxKind::MODULE => SymbolKind::Module,
        _ => SymbolKind::Variable,
    }
}

pub fn publish_diagnostics(world: World, uri: Url) -> Result<req::PublishDiagnosticsParams> {
    let path = uri.file_path()?;
    let file = world.file_syntax(&path)?;
    let line_index = world.file_line_index(&path)?;
    let diagnostics = libeditor::diagnostics(&file)
        .into_iter()
        .map(|d| Diagnostic {
            range: to_vs_range(&line_index, d.range),
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
            range: to_vs_range(&line_index, h.range),
            tag: h.tag,
        }).collect();
    Ok(req::PublishDecorationsParams { uri, decorations })
}

fn to_text_range(line_index: &LineIndex, range: Range) -> TextRange {
    TextRange::from_to(
        to_text_unit(line_index, range.start),
        to_text_unit(line_index, range.end),
    )
}

fn to_text_unit(line_index: &LineIndex, position: Position) -> TextUnit {
    // TODO: UTF-16
    let line_col = LineCol {
        line: position.line as u32,
        col: (position.character as u32).into(),
    };
    line_index.offset(line_col)
}


fn to_vs_range(line_index: &LineIndex, range: TextRange) -> Range {
    Range::new(
        to_vs_position(line_index, range.start()),
        to_vs_position(line_index, range.end()),
    )
}

fn to_vs_position(line_index: &LineIndex, offset: TextUnit) -> Position {
    let line_col = line_index.line_col(offset);
    // TODO: UTF-16
    Position::new(line_col.line as u64, u32::from(line_col.col) as u64)
}
