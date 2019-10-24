//! We use code generation heavily in rust-analyzer.
//!
//! Rather then doing it via proc-macros, we use old-school way of just dumping
//! the source code.
//!
//! This module's submodules define specific bits that we generate.

mod gen_syntax;
mod gen_parser_tests;

use std::{fs, mem, path::Path};

use crate::Result;

pub use self::{gen_parser_tests::generate_parser_tests, gen_syntax::generate_syntax};

pub const GRAMMAR: &str = "crates/ra_syntax/src/grammar.ron";
const GRAMMAR_DIR: &str = "crates/ra_parser/src/grammar";
const OK_INLINE_TESTS_DIR: &str = "crates/ra_syntax/test_data/parser/inline/ok";
const ERR_INLINE_TESTS_DIR: &str = "crates/ra_syntax/test_data/parser/inline/err";

pub const SYNTAX_KINDS: &str = "crates/ra_parser/src/syntax_kind/generated.rs";
pub const AST: &str = "crates/ra_syntax/src/ast/generated.rs";

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Mode {
    Overwrite,
    Verify,
}

/// A helper to update file on disk if it has changed.
/// With verify = false,
pub fn update(path: &Path, contents: &str, mode: Mode) -> Result<()> {
    match fs::read_to_string(path) {
        Ok(ref old_contents) if old_contents == contents => {
            return Ok(());
        }
        _ => (),
    }
    if mode == Mode::Verify {
        Err(format!("`{}` is not up-to-date", path.display()))?;
    }
    eprintln!("updating {}", path.display());
    fs::write(path, contents)?;
    Ok(())
}

fn extract_comment_blocks(text: &str) -> Vec<Vec<String>> {
    let mut res = Vec::new();

    let prefix = "// ";
    let lines = text.lines().map(str::trim_start);

    let mut block = vec![];
    for line in lines {
        let is_comment = line.starts_with(prefix);
        if is_comment {
            block.push(line[prefix.len()..].to_string());
        } else {
            if !block.is_empty() {
                res.push(mem::replace(&mut block, Vec::new()))
            }
        }
    }
    if !block.is_empty() {
        res.push(mem::replace(&mut block, Vec::new()))
    }
    res
}
