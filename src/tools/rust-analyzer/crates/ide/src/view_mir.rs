use hir::{DefWithBody, Semantics};
use ide_db::{FilePosition, RootDatabase};
use syntax::{AstNode, algo::ancestors_at_offset, ast};

// Feature: View Mir
//
// | Editor  | Action Name |
// |---------|-------------|
// | VS Code | **rust-analyzer: View Mir**
pub(crate) fn view_mir(db: &RootDatabase, position: FilePosition) -> String {
    body_mir(db, position).unwrap_or_else(|| "Not inside a function body".to_owned())
}

fn body_mir(db: &RootDatabase, position: FilePosition) -> Option<String> {
    let sema = Semantics::new(db);
    let source_file = sema.parse_guess_edition(position.file_id);

    let item = ancestors_at_offset(source_file.syntax(), position.offset)
        .filter(|it| !ast::MacroCall::can_cast(it.kind()))
        .find_map(ast::Item::cast)?;
    let def: DefWithBody = match item {
        ast::Item::Fn(it) => sema.to_def(&it)?.into(),
        ast::Item::Const(it) => sema.to_def(&it)?.into(),
        ast::Item::Static(it) => sema.to_def(&it)?.into(),
        _ => return None,
    };
    Some(def.debug_mir(db))
}
