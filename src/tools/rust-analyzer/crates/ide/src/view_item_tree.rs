use hir::{Semantics, db::DefDatabase};
use ide_db::{FileId, RootDatabase};

// Feature: Debug ItemTree
//
// Displays the ItemTree of the currently open file, for debugging.
//
// | Editor  | Action Name |
// |---------|-------------|
// | VS Code | **rust-analyzer: Debug ItemTree** |
pub(crate) fn view_item_tree(db: &RootDatabase, file_id: FileId) -> String {
    let sema = Semantics::new(db);
    let Some(krate) = sema.first_crate(file_id) else {
        return String::new();
    };
    let file_id = sema.attach_first_edition(file_id);
    db.file_item_tree(file_id.into(), krate.into()).pretty_print(db, file_id.edition(db))
}
