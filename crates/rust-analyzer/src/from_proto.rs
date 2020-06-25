//! Conversion lsp_types types to rust-analyzer specific ones.
use std::convert::TryFrom;

use ra_db::{FileId, FilePosition, FileRange};
use ra_ide::{LineCol, LineIndex};
use ra_syntax::{TextRange, TextSize};
use vfs::AbsPathBuf;

use crate::{global_state::GlobalStateSnapshot, Result};

pub(crate) fn abs_path(url: &lsp_types::Url) -> Result<AbsPathBuf> {
    let path = url.to_file_path().map_err(|()| "url is not a file")?;
    Ok(AbsPathBuf::try_from(path).unwrap())
}

pub(crate) fn vfs_path(url: &lsp_types::Url) -> Result<vfs::VfsPath> {
    abs_path(url).map(vfs::VfsPath::from)
}

pub(crate) fn offset(line_index: &LineIndex, position: lsp_types::Position) -> TextSize {
    let line_col = LineCol { line: position.line as u32, col_utf16: position.character as u32 };
    line_index.offset(line_col)
}

pub(crate) fn text_range(line_index: &LineIndex, range: lsp_types::Range) -> TextRange {
    let start = offset(line_index, range.start);
    let end = offset(line_index, range.end);
    TextRange::new(start, end)
}

pub(crate) fn file_id(world: &GlobalStateSnapshot, url: &lsp_types::Url) -> Result<FileId> {
    world.url_to_file_id(url)
}

pub(crate) fn file_position(
    world: &GlobalStateSnapshot,
    tdpp: lsp_types::TextDocumentPositionParams,
) -> Result<FilePosition> {
    let file_id = file_id(world, &tdpp.text_document.uri)?;
    let line_index = world.analysis.file_line_index(file_id)?;
    let offset = offset(&*line_index, tdpp.position);
    Ok(FilePosition { file_id, offset })
}

pub(crate) fn file_range(
    world: &GlobalStateSnapshot,
    text_document_identifier: lsp_types::TextDocumentIdentifier,
    range: lsp_types::Range,
) -> Result<FileRange> {
    let file_id = file_id(world, &text_document_identifier.uri)?;
    let line_index = world.analysis.file_line_index(file_id)?;
    let range = text_range(&line_index, range);
    Ok(FileRange { file_id, range })
}
