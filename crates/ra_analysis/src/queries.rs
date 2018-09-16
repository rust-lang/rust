use std::sync::Arc;
use ra_syntax::File;
use ra_editor::LineIndex;
use {
    FileId,
    db::{Query, QueryCtx, QueryRegistry},
    symbol_index::SymbolIndex,
};

pub(crate) use db::{file_text, file_set};

pub(crate) fn file_syntax(ctx: QueryCtx, file_id: FileId) -> File {
    (&*ctx.get(FILE_SYNTAX, file_id)).clone()
}
pub(crate) fn file_lines(ctx: QueryCtx, file_id: FileId) -> Arc<LineIndex> {
    ctx.get(FILE_LINES, file_id)
}
pub(crate) fn file_symbols(ctx: QueryCtx, file_id: FileId) -> Arc<SymbolIndex> {
    ctx.get(FILE_SYMBOLS, file_id)
}

const FILE_SYNTAX: Query<FileId, File> = Query(16, |ctx, file_id: &FileId| {
    let text = file_text(ctx, *file_id);
    File::parse(&*text)
});
const FILE_LINES: Query<FileId, LineIndex> = Query(17, |ctx, file_id: &FileId| {
    let text = file_text(ctx, *file_id);
    LineIndex::new(&*text)
});
const FILE_SYMBOLS: Query<FileId, SymbolIndex> = Query(18, |ctx, file_id: &FileId| {
    let syntax = file_syntax(ctx, *file_id);
    SymbolIndex::for_file(*file_id, syntax)
});

pub(crate) fn register_queries(reg: &mut QueryRegistry) {
    reg.add(FILE_SYNTAX, "FILE_SYNTAX");
    reg.add(FILE_LINES, "FILE_LINES");
    reg.add(FILE_SYMBOLS, "FILE_SYMBOLS");
}
