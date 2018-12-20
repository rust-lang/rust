use std::sync::Arc;

use salsa::Database;
use ra_db::{FilesDatabase, CrateGraph, SyntaxDatabase};
use ra_syntax::{SmolStr, algo::visit::{visitor, Visitor}, ast::{self, AstNode}};
use relative_path::RelativePath;

use crate::{source_binder, mock::WORKSPACE, module::ModuleSourceNode};

use crate::{
    self as hir,
    db::HirDatabase,
    mock::MockDatabase,
};

fn infer_all_fns(fixture: &str) -> () {
    let (db, source_root) = MockDatabase::with_files(fixture);
    for &file_id in source_root.files.values() {
        let source_file = db.source_file(file_id);
        for fn_def in source_file.syntax().descendants().filter_map(ast::FnDef::cast) {
            let func = source_binder::function_from_source(&db, file_id, fn_def).unwrap().unwrap();
            let inference_result = func.infer(&db);
            for (syntax_ptr, ty) in &inference_result.type_for {
                let node = syntax_ptr.resolve(&source_file);
                eprintln!("{} '{}': {:?}", syntax_ptr.range(), node.text(), ty);
            }
        }
    }
}

#[test]
fn infer_smoke_test() {
    let text = "
        //- /lib.rs
        fn foo(x: u32, y: !) -> i128 {
            x;
            y;
            return 1;
            \"hello\";
            0
        }
    ";

    infer_all_fns(text);
}
