//! We use code generation heavily in rust-analyzer.
//!
//! Rather then doing it via proc-macros, we use old-school way of just dumping
//! the source code.
//!
//! This module's submodules define specific bits that we generate.

mod gen_syntax;
mod gen_parser_tests;
mod gen_assists_docs;

use std::{fs, mem, path::Path};

use crate::Result;

pub use self::{
    gen_assists_docs::generate_assists_docs, gen_parser_tests::generate_parser_tests,
    gen_syntax::generate_syntax,
};

const GRAMMAR_DIR: &str = "crates/ra_parser/src/grammar";
const OK_INLINE_TESTS_DIR: &str = "crates/ra_syntax/test_data/parser/inline/ok";
const ERR_INLINE_TESTS_DIR: &str = "crates/ra_syntax/test_data/parser/inline/err";

pub const SYNTAX_KINDS: &str = "crates/ra_parser/src/syntax_kind/generated.rs";
pub const AST: &str = "crates/ra_syntax/src/ast/generated.rs";

const ASSISTS_DIR: &str = "crates/ra_assists/src/assists";
const ASSISTS_TESTS: &str = "crates/ra_assists/src/doc_tests/generated.rs";
const ASSISTS_DOCS: &str = "docs/user/assists.md";

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Mode {
    Overwrite,
    Verify,
}

/// A helper to update file on disk if it has changed.
/// With verify = false,
fn update(path: &Path, contents: &str, mode: Mode) -> Result<()> {
    match fs::read_to_string(path) {
        Ok(ref old_contents) if normalize(old_contents) == normalize(contents) => {
            return Ok(());
        }
        _ => (),
    }
    if mode == Mode::Verify {
        anyhow::bail!("`{}` is not up-to-date", path.display());
    }
    eprintln!("updating {}", path.display());
    fs::write(path, contents)?;
    return Ok(());

    fn normalize(s: &str) -> String {
        s.replace("\r\n", "\n")
    }
}

fn extract_comment_blocks(text: &str) -> Vec<Vec<String>> {
    do_extract_comment_blocks(text, false)
}

fn extract_comment_blocks_with_empty_lines(text: &str) -> Vec<Vec<String>> {
    do_extract_comment_blocks(text, true)
}

fn do_extract_comment_blocks(text: &str, allow_blocks_with_empty_lins: bool) -> Vec<Vec<String>> {
    let mut res = Vec::new();

    let prefix = "// ";
    let lines = text.lines().map(str::trim_start);

    let mut block = vec![];
    for line in lines {
        if line == "//" && allow_blocks_with_empty_lins {
            block.push(String::new());
            continue;
        }

        let is_comment = line.starts_with(prefix);
        if is_comment {
            block.push(line[prefix.len()..].to_string());
        } else if !block.is_empty() {
            res.push(mem::replace(&mut block, Vec::new()));
        }
    }
    if !block.is_empty() {
        res.push(mem::replace(&mut block, Vec::new()))
    }
    res
}
