use hir::{DefWithBody, Semantics};
use ide_db::base_db::FilePosition;
use ide_db::RootDatabase;
use syntax::{algo::ancestors_at_offset, ast, AstNode};

// Feature: View Hir
//
// |===
// | Editor  | Action Name
//
// | VS Code | **rust-analyzer: View Hir**
// |===
// image::https://user-images.githubusercontent.com/48062697/113065588-068bdb80-91b1-11eb-9a78-0b4ef1e972fb.gif[]
pub(crate) fn view_hir(db: &RootDatabase, position: FilePosition) -> String {
    body_hir(db, position).unwrap_or_else(|| "Not inside a function body".to_owned())
}

fn body_hir(db: &RootDatabase, position: FilePosition) -> Option<String> {
    let sema = Semantics::new(db);
    let source_file = sema.parse(position.file_id);

    let item = ancestors_at_offset(source_file.syntax(), position.offset)
        .filter(|it| !ast::MacroCall::can_cast(it.kind()))
        .find_map(ast::Item::cast)?;
    let def: DefWithBody = match item {
        ast::Item::Fn(it) => sema.to_def(&it)?.into(),
        ast::Item::Const(it) => sema.to_def(&it)?.into(),
        ast::Item::Static(it) => sema.to_def(&it)?.into(),
        _ => return None,
    };
    Some(def.debug_hir(db))
}
