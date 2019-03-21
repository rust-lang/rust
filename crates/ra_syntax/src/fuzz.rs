use crate::{SourceFile, validation, AstNode};

fn check_file_invariants(file: &SourceFile) {
    let root = file.syntax();
    validation::validate_block_structure(root);
    let _ = file.errors();
}

pub fn check_parser(text: &str) {
    let file = SourceFile::parse(text);
    check_file_invariants(&file);
}
