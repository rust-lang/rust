//! Conversion lsp_types types to rust-analyzer specific ones.
use anyhow::format_err;
use ide::{Annotation, AnnotationKind, AssistKind, LineCol};
use ide_db::{FileId, FilePosition, FileRange, line_index::WideLineCol};
use paths::Utf8PathBuf;
use syntax::{TextRange, TextSize};
use vfs::AbsPathBuf;

use crate::{
    global_state::GlobalStateSnapshot,
    line_index::{LineIndex, PositionEncoding},
    lsp_ext, try_default,
};

pub(crate) fn abs_path(url: &lsp_types::Url) -> anyhow::Result<AbsPathBuf> {
    let path = url.to_file_path().map_err(|()| anyhow::format_err!("url is not a file"))?;
    Ok(AbsPathBuf::try_from(Utf8PathBuf::from_path_buf(path).unwrap()).unwrap())
}

pub(crate) fn vfs_path(url: &lsp_types::Url) -> anyhow::Result<vfs::VfsPath> {
    abs_path(url).map(vfs::VfsPath::from)
}

pub(crate) fn offset(
    line_index: &LineIndex,
    position: lsp_types::Position,
) -> anyhow::Result<TextSize> {
    let line_col = match line_index.encoding {
        PositionEncoding::Utf8 => LineCol { line: position.line, col: position.character },
        PositionEncoding::Wide(enc) => {
            let line_col = WideLineCol { line: position.line, col: position.character };
            line_index
                .index
                .to_utf8(enc, line_col)
                .ok_or_else(|| format_err!("Invalid wide col offset"))?
        }
    };
    let line_range = line_index.index.line(line_col.line).ok_or_else(|| {
        format_err!("Invalid offset {line_col:?} (line index length: {:?})", line_index.index.len())
    })?;
    let col = TextSize::from(line_col.col);
    let clamped_len = col.min(line_range.len());
    if clamped_len < col {
        tracing::error!(
            "Position {line_col:?} column exceeds line length {}, clamping it",
            u32::from(line_range.len()),
        );
    }
    Ok(line_range.start() + clamped_len)
}

pub(crate) fn text_range(
    line_index: &LineIndex,
    range: lsp_types::Range,
) -> anyhow::Result<TextRange> {
    let start = offset(line_index, range.start)?;
    let end = offset(line_index, range.end)?;
    match end < start {
        true => Err(format_err!("Invalid Range")),
        false => Ok(TextRange::new(start, end)),
    }
}

/// Returns `None` if the file was excluded.
pub(crate) fn file_id(
    snap: &GlobalStateSnapshot,
    url: &lsp_types::Url,
) -> anyhow::Result<Option<FileId>> {
    snap.url_to_file_id(url)
}

/// Returns `None` if the file was excluded.
pub(crate) fn file_position(
    snap: &GlobalStateSnapshot,
    tdpp: lsp_types::TextDocumentPositionParams,
) -> anyhow::Result<Option<FilePosition>> {
    let file_id = try_default!(file_id(snap, &tdpp.text_document.uri)?);
    let line_index = snap.file_line_index(file_id)?;
    let offset = offset(&line_index, tdpp.position)?;
    Ok(Some(FilePosition { file_id, offset }))
}

/// Returns `None` if the file was excluded.
pub(crate) fn file_range(
    snap: &GlobalStateSnapshot,
    text_document_identifier: &lsp_types::TextDocumentIdentifier,
    range: lsp_types::Range,
) -> anyhow::Result<Option<FileRange>> {
    file_range_uri(snap, &text_document_identifier.uri, range)
}

/// Returns `None` if the file was excluded.
pub(crate) fn file_range_uri(
    snap: &GlobalStateSnapshot,
    document: &lsp_types::Url,
    range: lsp_types::Range,
) -> anyhow::Result<Option<FileRange>> {
    let file_id = try_default!(file_id(snap, document)?);
    let line_index = snap.file_line_index(file_id)?;
    let range = text_range(&line_index, range)?;
    Ok(Some(FileRange { file_id, range }))
}

pub(crate) fn assist_kind(kind: lsp_types::CodeActionKind) -> Option<AssistKind> {
    let assist_kind = match &kind {
        k if k == &lsp_types::CodeActionKind::EMPTY => AssistKind::Generate,
        k if k == &lsp_types::CodeActionKind::QUICKFIX => AssistKind::QuickFix,
        k if k == &lsp_types::CodeActionKind::REFACTOR => AssistKind::Refactor,
        k if k == &lsp_types::CodeActionKind::REFACTOR_EXTRACT => AssistKind::RefactorExtract,
        k if k == &lsp_types::CodeActionKind::REFACTOR_INLINE => AssistKind::RefactorInline,
        k if k == &lsp_types::CodeActionKind::REFACTOR_REWRITE => AssistKind::RefactorRewrite,
        _ => return None,
    };

    Some(assist_kind)
}

/// Returns `None` if the file was excluded.
pub(crate) fn annotation(
    snap: &GlobalStateSnapshot,
    range: lsp_types::Range,
    data: lsp_ext::CodeLensResolveData,
) -> anyhow::Result<Option<Annotation>> {
    match data.kind {
        lsp_ext::CodeLensResolveDataKind::Impls(params) => {
            if snap.url_file_version(&params.text_document_position_params.text_document.uri)
                != Some(data.version)
            {
                return Ok(None);
            }
            let pos @ FilePosition { file_id, .. } =
                try_default!(file_position(snap, params.text_document_position_params)?);
            let line_index = snap.file_line_index(file_id)?;

            Ok(Annotation {
                range: text_range(&line_index, range)?,
                kind: AnnotationKind::HasImpls { pos, data: None },
            })
        }
        lsp_ext::CodeLensResolveDataKind::References(params) => {
            if snap.url_file_version(&params.text_document.uri) != Some(data.version) {
                return Ok(None);
            }
            let pos @ FilePosition { file_id, .. } = try_default!(file_position(snap, params)?);
            let line_index = snap.file_line_index(file_id)?;

            Ok(Annotation {
                range: text_range(&line_index, range)?,
                kind: AnnotationKind::HasReferences { pos, data: None },
            })
        }
    }
    .map(Some)
}
