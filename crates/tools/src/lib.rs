use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    fs::copy,
    io::{Error, ErrorKind}
};

use failure::bail;
use itertools::Itertools;

pub use teraron::{Mode, Overwrite, Verify};

pub type Result<T> = std::result::Result<T, failure::Error>;

pub const GRAMMAR: &str = "crates/ra_syntax/src/grammar.ron";
pub const SYNTAX_KINDS: &str = "crates/ra_syntax/src/syntax_kinds/generated.rs.tera";
pub const AST: &str = "crates/ra_syntax/src/ast/generated.rs.tera";
const TOOLCHAIN: &str = "stable";

#[derive(Debug)]
pub struct Test {
    pub name: String,
    pub text: String,
    pub ok: bool,
}

pub fn collect_tests(s: &str) -> Vec<(usize, Test)> {
    let mut res = vec![];
    let prefix = "// ";
    let comment_blocks = s
        .lines()
        .map(str::trim_start)
        .enumerate()
        .group_by(|(_idx, line)| line.starts_with(prefix));

    'outer: for (is_comment, block) in comment_blocks.into_iter() {
        if !is_comment {
            continue;
        }
        let mut block = block.map(|(idx, line)| (idx, &line[prefix.len()..]));

        let mut ok = true;
        let (start_line, name) = loop {
            match block.next() {
                Some((idx, line)) if line.starts_with("test ") => {
                    break (idx, line["test ".len()..].to_string());
                }
                Some((idx, line)) if line.starts_with("test_err ") => {
                    ok = false;
                    break (idx, line["test_err ".len()..].to_string());
                }
                Some(_) => (),
                None => continue 'outer,
            }
        };
        let text: String = itertools::join(
            block.map(|(_, line)| line).chain(::std::iter::once("")),
            "\n",
        );
        assert!(!text.trim().is_empty() && text.ends_with('\n'));
        res.push((start_line, Test { name, text, ok }))
    }
    res
}

pub fn generate(mode: Mode) -> Result<()> {
    let grammar = project_root().join(GRAMMAR);
    let syntax_kinds = project_root().join(SYNTAX_KINDS);
    let ast = project_root().join(AST);
    teraron::generate(&syntax_kinds, &grammar, mode)?;
    teraron::generate(&ast, &grammar, mode)?;
    Ok(())
}

pub fn project_root() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

pub fn run(cmdline: &str, dir: &str) -> Result<()> {
    eprintln!("\nwill run: {}", cmdline);
    let project_dir = project_root().join(dir);
    let mut args = cmdline.split_whitespace();
    let exec = args.next().unwrap();
    let status = Command::new(exec)
        .args(args)
        .current_dir(project_dir)
        .status()?;
    if !status.success() {
        bail!("`{}` exited with {}", cmdline, status);
    }
    Ok(())
}

pub fn run_rustfmt(mode: Mode) -> Result<()> {
    match Command::new("rustup")
        .args(&["run", TOOLCHAIN, "--", "cargo", "fmt", "--version"])
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
    {
        Ok(status) if status.success() => (),
        _ => install_rustfmt()?,
    };

    if mode == Verify {
        run(
            &format!("rustup run {} -- cargo fmt -- --check", TOOLCHAIN),
            ".",
        )?;
    } else {
        run(&format!("rustup run {} -- cargo fmt", TOOLCHAIN), ".")?;
    }
    Ok(())
}

pub fn install_rustfmt() -> Result<()> {
    run(&format!("rustup install {}", TOOLCHAIN), ".")?;
    run(
        &format!("rustup component add rustfmt --toolchain {}", TOOLCHAIN),
        ".",
    )
}

pub fn install_format_hook() -> Result<()> {
    let result_path = Path::new("./.git/hooks/pre-commit");
    if !result_path.exists() {
        run("cargo build --package tools --bin pre-commit", ".")?;
        if cfg!(windows) {
            copy("./target/debug/pre-commit.exe", result_path)?;
        } else {
            copy("./target/debug/pre-commit", result_path)?;
        }
    } else {
        return Err(Error::new(ErrorKind::AlreadyExists, "Git hook already created").into());
    }
    Ok(())
}

pub fn run_fuzzer() -> Result<()> {
    match Command::new("cargo")
        .args(&["fuzz", "--help"])
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
    {
        Ok(status) if status.success() => (),
        _ => run("cargo install cargo-fuzz", ".")?,
    };

    run(
        "rustup run nightly -- cargo fuzz run parser",
        "./crates/ra_syntax",
    )
}
