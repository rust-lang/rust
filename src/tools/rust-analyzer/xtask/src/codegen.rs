use std::{
    fmt, fs, mem,
    path::{Path, PathBuf},
};

use xshell::{Shell, cmd};

use crate::{
    flags::{self, CodegenType},
    project_root,
};

pub(crate) mod assists_doc_tests;
pub(crate) mod diagnostics_docs;
pub(crate) mod feature_docs;
mod grammar;
mod lints;
mod parser_inline_tests;

impl flags::Codegen {
    pub(crate) fn run(self, _sh: &Shell) -> anyhow::Result<()> {
        match self.codegen_type.unwrap_or_default() {
            flags::CodegenType::All => {
                grammar::generate(self.check);
                assists_doc_tests::generate(self.check);
                parser_inline_tests::generate(self.check);
                feature_docs::generate(self.check);
                diagnostics_docs::generate(self.check);
                // lints::generate(self.check) Updating clones the rust repo, so don't run it unless
                // explicitly asked for
            }
            flags::CodegenType::Grammar => grammar::generate(self.check),
            flags::CodegenType::AssistsDocTests => assists_doc_tests::generate(self.check),
            flags::CodegenType::DiagnosticsDocs => diagnostics_docs::generate(self.check),
            flags::CodegenType::LintDefinitions => lints::generate(self.check),
            flags::CodegenType::ParserTests => parser_inline_tests::generate(self.check),
            flags::CodegenType::FeatureDocs => feature_docs::generate(self.check),
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct CommentBlock {
    pub(crate) id: String,
    pub(crate) line: usize,
    pub(crate) contents: Vec<String>,
    is_doc: bool,
}

impl CommentBlock {
    fn extract(tag: &str, text: &str) -> Vec<CommentBlock> {
        assert!(tag.starts_with(char::is_uppercase));

        let tag = format!("{tag}:");
        let mut blocks = CommentBlock::extract_untagged(text);
        blocks.retain_mut(|block| {
            let first = block.contents.remove(0);
            let Some(id) = first.strip_prefix(&tag) else {
                return false;
            };

            if block.is_doc {
                panic!("Use plain (non-doc) comments with tags like {tag}:\n    {first}");
            }

            id.trim().clone_into(&mut block.id);
            true
        });
        blocks
    }

    fn extract_untagged(text: &str) -> Vec<CommentBlock> {
        let mut res = Vec::new();

        let lines = text.lines().map(str::trim_start);

        let dummy_block =
            CommentBlock { id: String::new(), line: 0, contents: Vec::new(), is_doc: false };
        let mut block = dummy_block.clone();
        for (line_num, line) in lines.enumerate() {
            match line.strip_prefix("//") {
                Some(mut contents) if !contents.starts_with('/') => {
                    if let Some('/' | '!') = contents.chars().next() {
                        contents = &contents[1..];
                        block.is_doc = true;
                    }
                    if let Some(' ') = contents.chars().next() {
                        contents = &contents[1..];
                    }
                    block.contents.push(contents.to_owned());
                }
                _ => {
                    if !block.contents.is_empty() {
                        let block = mem::replace(&mut block, dummy_block.clone());
                        res.push(block);
                    }
                    block.line = line_num + 2;
                }
            }
        }
        if !block.contents.is_empty() {
            res.push(block);
        }
        res
    }
}

#[derive(Debug)]
pub(crate) struct Location {
    pub(crate) file: PathBuf,
    pub(crate) line: usize,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = self.file.strip_prefix(project_root()).unwrap().display().to_string();
        let path = path.replace('\\', "/");
        let name = self.file.file_name().unwrap();
        write!(
            f,
            " [{}](https://github.com/rust-lang/rust-analyzer/blob/master/{}#L{}) ",
            name.to_str().unwrap(),
            path,
            self.line
        )
    }
}

fn reformat(text: String) -> String {
    let sh = Shell::new().unwrap();
    let rustfmt_toml = project_root().join("rustfmt.toml");
    let version = cmd!(sh, "rustup run stable rustfmt --version").read().unwrap_or_default();

    // First try explicitly requesting the stable channel via rustup in case nightly is being used by default,
    // then plain rustfmt in case rustup isn't being used to manage the compiler (e.g. when using Nix).
    let mut stdout = if !version.contains("stable") {
        let version = cmd!(sh, "rustfmt --version").read().unwrap_or_default();
        if !version.contains("stable") {
            panic!(
                "Failed to run rustfmt from toolchain 'stable'. \
                 Please run `rustup component add rustfmt --toolchain stable` to install it.",
            );
        } else {
            cmd!(sh, "rustfmt --config-path {rustfmt_toml} --config fn_single_line=true")
                .stdin(text)
                .read()
                .unwrap()
        }
    } else {
        cmd!(
            sh,
            "rustup run stable rustfmt --config-path {rustfmt_toml} --config fn_single_line=true"
        )
        .stdin(text)
        .read()
        .unwrap()
    };
    if !stdout.ends_with('\n') {
        stdout.push('\n');
    }
    stdout
}

fn add_preamble(cg: CodegenType, mut text: String) -> String {
    let preamble = format!("//! Generated by `cargo xtask codegen {cg}`, do not edit by hand.\n\n");
    text.insert_str(0, &preamble);
    text
}

/// Checks that the `file` has the specified `contents`. If that is not the
/// case, updates the file and then fails the test.
#[allow(clippy::print_stderr)]
fn ensure_file_contents(cg: CodegenType, file: &Path, contents: &str, check: bool) -> bool {
    let contents = normalize_newlines(contents);
    if let Ok(old_contents) = fs::read_to_string(file)
        && normalize_newlines(&old_contents) == contents
    {
        // File is already up to date.
        return false;
    }

    let display_path = file.strip_prefix(project_root()).unwrap_or(file);
    if check {
        panic!(
            "{} was not up-to-date{}",
            file.display(),
            if std::env::var("CI").is_ok() {
                format!(
                    "\n    NOTE: run `cargo xtask codegen {cg}` locally and commit the updated files\n"
                )
            } else {
                "".to_owned()
            }
        );
    } else {
        eprintln!(
            "\n\x1b[31;1merror\x1b[0m: {} was not up-to-date, updating\n",
            display_path.display()
        );

        if let Some(parent) = file.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(file, contents).unwrap();
        true
    }
}

fn normalize_newlines(s: &str) -> String {
    s.replace("\r\n", "\n")
}
