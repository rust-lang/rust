use hir::{DefWithBody, Semantics};
use ide_db::base_db::FilePosition;
use ide_db::RootDatabase;
use syntax::{algo::find_node_at_offset, ast, AstNode};

// Feature: View Mir
//
// |===
// | Editor  | Action Name
//
// | VS Code | **rust-analyzer: View Mir**
// |===
pub(crate) fn view_mir(db: &RootDatabase, position: FilePosition) -> String {
    body_mir(db, position).unwrap_or_else(|| "Not inside a function body".to_string())
}

fn body_mir(db: &RootDatabase, position: FilePosition) -> Option<String> {
    let sema = Semantics::new(db);
    let source_file = sema.parse(position.file_id);

    let item = find_node_at_offset::<ast::Item>(source_file.syntax(), position.offset)?;
    let def: DefWithBody = match item {
        ast::Item::Fn(it) => sema.to_def(&it)?.into(),
        ast::Item::Const(it) => sema.to_def(&it)?.into(),
        ast::Item::Static(it) => sema.to_def(&it)?.into(),
        _ => return None,
    };
    Some(def.debug_mir(db))
}
