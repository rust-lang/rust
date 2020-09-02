use hir::Semantics;
use ide_db::RootDatabase;
use syntax::{ast, ast::NameOwner, AstNode, SyntaxNode};

use crate::{runnables::has_test_related_attribute, FileId, FileRange};

pub(crate) fn find_all_methods(db: &RootDatabase, file_id: FileId) -> Vec<FileRange> {
    let sema = Semantics::new(db);
    let source_file = sema.parse(file_id);
    source_file.syntax().descendants().filter_map(|it| method_range(it, file_id)).collect()
}

pub(crate) fn method_range(item: SyntaxNode, file_id: FileId) -> Option<FileRange> {
    ast::Fn::cast(item).and_then(|fn_def|{
        if has_test_related_attribute(&fn_def) {
            None
        } else {
            fn_def.name().map(|name| FileRange{ file_id, range: name.syntax().text_range() })
        }
    })
}
