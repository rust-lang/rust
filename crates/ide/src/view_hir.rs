use hir::{Function, Semantics};
use hir::db::DefDatabase;
use ide_db::base_db::FilePosition;
use ide_db::RootDatabase;
use syntax::{AstNode, algo::find_node_at_offset, ast};
use std::fmt::Write;

// Feature: View hir
//
// |===
// | Editor  | Action Name
//
// | VS Code | **Rust Analyzer: View Hir**
// |===
pub(crate) fn view_hir(db: &RootDatabase, position: FilePosition) -> String {
    body_hir(db, position).unwrap_or("Not inside a function body".to_string())
}

fn body_hir(db: &RootDatabase, position: FilePosition) -> Option<String> {
    let sema = Semantics::new(db);
    let source_file = sema.parse(position.file_id);

    let function = find_node_at_offset::<ast::Fn>(
        source_file.syntax(),
        position.offset,
    )?;

    let function: Function = sema.to_def(&function)?;
    let body = db.body(function.id.into());

    let mut result = String::new();
    writeln!(&mut result, "== Body expressions ==").ok()?;

    for (id, expr) in body.exprs.iter() {
        writeln!(&mut result, "{:?}: {:?}", id, expr).ok()?;
    }

    Some(result)
}