use std::fmt::Write;
use std::sync::Arc;
use std::path::{Path, PathBuf};

use salsa::Database;
use ra_db::{FilesDatabase, CrateGraph, SyntaxDatabase};
use ra_syntax::{SmolStr, algo::visit::{visitor, Visitor}, ast::{self, AstNode}};
use test_utils::{project_dir, dir_tests};
use relative_path::RelativePath;

use crate::{source_binder, mock::WORKSPACE, module::ModuleSourceNode};

use crate::{
    self as hir,
    db::HirDatabase,
    mock::MockDatabase,
};

fn infer_file(content: &str) -> String {
    let (db, source_root, file_id) = MockDatabase::with_single_file(content);
    let source_file = db.source_file(file_id);
    let mut acc = String::new();
    for fn_def in source_file.syntax().descendants().filter_map(ast::FnDef::cast) {
        let func = source_binder::function_from_source(&db, file_id, fn_def).unwrap().unwrap();
        let inference_result = func.infer(&db);
        for (syntax_ptr, ty) in &inference_result.type_for {
            let node = syntax_ptr.resolve(&source_file);
            write!(acc, "{} '{}': {}\n", syntax_ptr.range(), ellipsize(node.text().to_string().replace("\n", " "), 15), ty);
        }
    }
    acc
}

fn ellipsize(mut text: String, max_len: usize) -> String {
    if text.len() <= max_len {
        return text;
    }
    let ellipsis = "...";
    let e_len = ellipsis.len();
    let mut prefix_len = (max_len - e_len) / 2;
    while !text.is_char_boundary(prefix_len) {
        prefix_len += 1;
    }
    let mut suffix_len = max_len - e_len - prefix_len;
    while !text.is_char_boundary(text.len() - suffix_len) {
        suffix_len += 1;
    }
    text.replace_range(prefix_len..text.len() - suffix_len, ellipsis);
    text
}

#[test]
pub fn infer_tests() {
    dir_tests(&test_data_dir(), &["."], |text, _path| {
        infer_file(text)
    });
}

fn test_data_dir() -> PathBuf {
    project_dir().join("crates/ra_hir/src/ty/tests/data")
}
