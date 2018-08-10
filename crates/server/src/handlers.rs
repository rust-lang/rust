use languageserver_types::{Range, Position};
use libanalysis::World;
use libeditor::{self, LineIndex, LineCol, TextRange, TextUnit};
use {req, Result, FilePath};

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
