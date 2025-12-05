use hir::Semantics;
use ide_db::{FilePosition, RootDatabase};
use syntax::AstNode;

// Feature: View Hir
//
// | Editor  | Action Name |
// |---------|--------------|
// | VS Code | **rust-analyzer: View Hir**
//
// ![View Hir](https://user-images.githubusercontent.com/48062697/113065588-068bdb80-91b1-11eb-9a78-0b4ef1e972fb.gif)
pub(crate) fn view_hir(db: &RootDatabase, position: FilePosition) -> String {
    (|| {
        let sema = Semantics::new(db);
        let source_file = sema.parse_guess_edition(position.file_id);
        sema.debug_hir_at(source_file.syntax().token_at_offset(position.offset).next()?)
    })()
    .unwrap_or_else(|| "Not inside a lowerable item".to_owned())
}
