use std::sync::Arc;
use libsyntax2::File;
use libeditor::LineIndex;
use {
    FileId,
    db::{Query, QueryCtx, QueryRegistry, file_text},
};

pub(crate) fn register_queries(reg: &mut QueryRegistry) {
    reg.add(FILE_SYNTAX, "FILE_SYNTAX");
    reg.add(FILE_LINES, "FILE_LINES");
}

pub(crate) fn file_syntax(ctx: QueryCtx, file_id: FileId) -> File {
    (&*ctx.get(FILE_SYNTAX, file_id)).clone()
}
pub(crate) fn file_lines(ctx: QueryCtx, file_id: FileId) -> Arc<LineIndex> {
    ctx.get(FILE_LINES, file_id)
}

const FILE_SYNTAX: Query<FileId, File> = Query(16, |ctx, file_id: &FileId| {
    let text = file_text(ctx, *file_id);
    File::parse(&*text)
});
const FILE_LINES: Query<FileId, LineIndex> = Query(17, |ctx, file_id: &FileId| {
    let text = file_text(ctx, *file_id);
    LineIndex::new(&*text)
});
