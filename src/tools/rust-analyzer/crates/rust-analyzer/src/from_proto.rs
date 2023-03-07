//! Conversion lsp_types types to rust-analyzer specific ones.
use anyhow::format_err;
use ide::{Annotation, AnnotationKind, AssistKind, LineCol};
use ide_db::{
    base_db::{FileId, FilePosition, FileRange},
    line_index::WideLineCol,
};
use syntax::{TextRange, TextSize};
use vfs::AbsPathBuf;

use crate::{
    from_json,
    global_state::GlobalStateSnapshot,
    line_index::{LineIndex, PositionEncoding},
    lsp_ext,
    lsp_utils::invalid_params_error,
    Result,
};

pub(crate) fn abs_path(url: &lsp_types::Url) -> Result<AbsPathBuf> {
    let path = url.to_file_path().map_err(|()| "url is not a file")?;
    Ok(AbsPathBuf::try_from(path).unwrap())
}

pub(crate) fn vfs_path(url: &lsp_types::Url) -> Result<vfs::VfsPath> {
    abs_path(url).map(vfs::VfsPath::from)
}

pub(crate) fn offset(line_index: &LineIndex, position: lsp_types::Position) -> Result<TextSize> {
    let line_col = match line_index.encoding {
        PositionEncoding::Utf8 => LineCol { line: position.line, col: position.character },
        PositionEncoding::Wide(enc) => {
            let line_col = WideLineCol { line: position.line, col: position.character };
            line_index.index.to_utf8(enc, line_col)
        }
    };
    let text_size =
        line_index.index.offset(line_col).ok_or_else(|| format_err!("Invalid offset"))?;
    Ok(text_size)
}

pub(crate) fn text_range(line_index: &LineIndex, range: lsp_types::Range) -> Result<TextRange> {
    let start = offset(line_index, range.start)?;
    let end = offset(line_index, range.end)?;
    match end < start {
        true => Err(format_err!("Invalid Range").into()),
        false => Ok(TextRange::new(start, end)),
    }
}

pub(crate) fn file_id(snap: &GlobalStateSnapshot, url: &lsp_types::Url) -> Result<FileId> {
    snap.url_to_file_id(url)
}

pub(crate) fn file_position(
    snap: &GlobalStateSnapshot,
    tdpp: lsp_types::TextDocumentPositionParams,
) -> Result<FilePosition> {
    let file_id = file_id(snap, &tdpp.text_document.uri)?;
    let line_index = snap.file_line_index(file_id)?;
    let offset = offset(&line_index, tdpp.position)?;
    Ok(FilePosition { file_id, offset })
}

pub(crate) fn file_range(
    snap: &GlobalStateSnapshot,
    text_document_identifier: lsp_types::TextDocumentIdentifier,
    range: lsp_types::Range,
) -> Result<FileRange> {
    file_range_uri(snap, &text_document_identifier.uri, range)
}

pub(crate) fn file_range_uri(
    snap: &GlobalStateSnapshot,
    document: &lsp_types::Url,
    range: lsp_types::Range,
) -> Result<FileRange> {
    let file_id = file_id(snap, document)?;
    let line_index = snap.file_line_index(file_id)?;
    let range = text_range(&line_index, range)?;
    Ok(FileRange { file_id, range })
}

pub(crate) fn assist_kind(kind: lsp_types::CodeActionKind) -> Option<AssistKind> {
    let assist_kind = match &kind {
        k if k == &lsp_types::CodeActionKind::EMPTY => AssistKind::None,
        k if k == &lsp_types::CodeActionKind::QUICKFIX => AssistKind::QuickFix,
        k if k == &lsp_types::CodeActionKind::REFACTOR => AssistKind::Refactor,
        k if k == &lsp_types::CodeActionKind::REFACTOR_EXTRACT => AssistKind::RefactorExtract,
        k if k == &lsp_types::CodeActionKind::REFACTOR_INLINE => AssistKind::RefactorInline,
        k if k == &lsp_types::CodeActionKind::REFACTOR_REWRITE => AssistKind::RefactorRewrite,
        _ => return None,
    };

    Some(assist_kind)
}

pub(crate) fn annotation(
    snap: &GlobalStateSnapshot,
    code_lens: lsp_types::CodeLens,
) -> Result<Annotation> {
    let data =
        code_lens.data.ok_or_else(|| invalid_params_error("code lens without data".to_string()))?;
    let resolve = from_json::<lsp_ext::CodeLensResolveData>("CodeLensResolveData", &data)?;

    match resolve {
        lsp_ext::CodeLensResolveData::Impls(params) => {
            let pos @ FilePosition { file_id, .. } =
                file_position(snap, params.text_document_position_params)?;
            let line_index = snap.file_line_index(file_id)?;

            Ok(Annotation {
                range: text_range(&line_index, code_lens.range)?,
                kind: AnnotationKind::HasImpls { pos, data: None },
            })
        }
        lsp_ext::CodeLensResolveData::References(params) => {
            let pos @ FilePosition { file_id, .. } = file_position(snap, params)?;
            let line_index = snap.file_line_index(file_id)?;

            Ok(Annotation {
                range: text_range(&line_index, code_lens.range)?,
                kind: AnnotationKind::HasReferences { pos, data: None },
            })
        }
    }
}
