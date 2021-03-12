//! We use code generation heavily in rust-analyzer.
//!
//! Rather then doing it via proc-macros, we use old-school way of just dumping
//! the source code.
//!
//! This module's submodules define specific bits that we generate.

mod gen_syntax;
mod gen_parser_tests;
mod gen_lint_completions;
mod gen_assists_docs;
mod gen_feature_docs;
mod gen_diagnostic_docs;

use std::{
    fmt, mem,
    path::{Path, PathBuf},
};
use xshell::{cmd, pushenv};

use crate::{ensure_rustfmt, project_root, Result};

pub(crate) use self::{
    gen_assists_docs::generate_assists_tests, gen_lint_completions::generate_lint_completions,
    gen_parser_tests::generate_parser_tests, gen_syntax::generate_syntax,
};

pub(crate) fn docs() -> Result<()> {
    // We don't commit docs to the repo, so we can just overwrite them.
    gen_assists_docs::generate_assists_docs()?;
    gen_feature_docs::generate_feature_docs()?;
    gen_diagnostic_docs::generate_diagnostic_docs()?;
    Ok(())
}

#[allow(unused)]
fn used() {
    generate_parser_tests();
    generate_assists_tests();
    generate_syntax();
    generate_lint_completions();
}

/// Checks that the `file` has the specified `contents`. If that is not the
/// case, updates the file and then fails the test.
pub(crate) fn ensure_file_contents(file: &Path, contents: &str) -> Result<()> {
    match std::fs::read_to_string(file) {
        Ok(old_contents) if normalize_newlines(&old_contents) == normalize_newlines(contents) => {
            return Ok(())
        }
        _ => (),
    }
    let display_path = file.strip_prefix(&project_root()).unwrap_or(file);
    eprintln!(
        "\n\x1b[31;1merror\x1b[0m: {} was not up-to-date, updating\n",
        display_path.display()
    );
    if std::env::var("CI").is_ok() {
        eprintln!("    NOTE: run `cargo test` locally and commit the updated files\n");
    }
    if let Some(parent) = file.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(file, contents).unwrap();
    anyhow::bail!("some file was not up to date and has been updated, simply re-run the tests")
}

fn normalize_newlines(s: &str) -> String {
    s.replace("\r\n", "\n")
}

const PREAMBLE: &str = "Generated file, do not edit by hand, see `xtask/src/codegen`";

fn reformat(text: &str) -> Result<String> {
    let _e = pushenv("RUSTUP_TOOLCHAIN", "stable");
    ensure_rustfmt()?;
    let rustfmt_toml = project_root().join("rustfmt.toml");
    let stdout = cmd!("rustfmt --config-path {rustfmt_toml} --config fn_single_line=true")
        .stdin(text)
        .read()?;
    Ok(format!("//! {}\n\n{}\n", PREAMBLE, stdout))
}

fn extract_comment_blocks(text: &str) -> Vec<Vec<String>> {
    do_extract_comment_blocks(text, false).into_iter().map(|(_line, block)| block).collect()
}

fn extract_comment_blocks_with_empty_lines(tag: &str, text: &str) -> Vec<CommentBlock> {
    assert!(tag.starts_with(char::is_uppercase));
    let tag = format!("{}:", tag);
    let mut res = Vec::new();
    for (line, mut block) in do_extract_comment_blocks(text, true) {
        let first = block.remove(0);
        if first.starts_with(&tag) {
            let id = first[tag.len()..].trim().to_string();
            let block = CommentBlock { id, line, contents: block };
            res.push(block);
        }
    }
    res
}

struct CommentBlock {
    id: String,
    line: usize,
    contents: Vec<String>,
}

fn do_extract_comment_blocks(
    text: &str,
    allow_blocks_with_empty_lines: bool,
) -> Vec<(usize, Vec<String>)> {
    let mut res = Vec::new();

    let prefix = "// ";
    let lines = text.lines().map(str::trim_start);

    let mut block = (0, vec![]);
    for (line_num, line) in lines.enumerate() {
        if line == "//" && allow_blocks_with_empty_lines {
            block.1.push(String::new());
            continue;
        }

        let is_comment = line.starts_with(prefix);
        if is_comment {
            block.1.push(line[prefix.len()..].to_string());
        } else {
            if !block.1.is_empty() {
                res.push(mem::take(&mut block));
            }
            block.0 = line_num + 2;
        }
    }
    if !block.1.is_empty() {
        res.push(block)
    }
    res
}

#[derive(Debug)]
struct Location {
    file: PathBuf,
    line: usize,
}

impl Location {
    fn new(file: PathBuf, line: usize) -> Self {
        Self { file, line }
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = self.file.strip_prefix(&project_root()).unwrap().display().to_string();
        let path = path.replace('\\', "/");
        let name = self.file.file_name().unwrap();
        write!(
            f,
            "https://github.com/rust-analyzer/rust-analyzer/blob/master/{}#L{}[{}]",
            path,
            self.line,
            name.to_str().unwrap()
        )
    }
}
