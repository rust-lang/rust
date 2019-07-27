use std::{
    collections::HashMap,
    error::Error,
    fs,
    io::{Error as IoError, ErrorKind},
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
};

use itertools::Itertools;

pub use teraron::{Mode, Overwrite, Verify};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub const GRAMMAR: &str = "crates/ra_syntax/src/grammar.ron";
const GRAMMAR_DIR: &str = "crates/ra_parser/src/grammar";
const OK_INLINE_TESTS_DIR: &str = "crates/ra_syntax/test_data/parser/inline/ok";
const ERR_INLINE_TESTS_DIR: &str = "crates/ra_syntax/test_data/parser/inline/err";

pub const SYNTAX_KINDS: &str = "crates/ra_parser/src/syntax_kind/generated.rs.tera";
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
        let text: String =
            itertools::join(block.map(|(_, line)| line).chain(::std::iter::once("")), "\n");
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
    Path::new(&env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().to_path_buf()
}

pub struct Cmd {
    pub unix: &'static str,
    pub windows: &'static str,
    pub work_dir: &'static str,
}

impl Cmd {
    pub fn run(self) -> Result<()> {
        if cfg!(windows) {
            run(self.windows, self.work_dir)
        } else {
            run(self.unix, self.work_dir)
        }
    }
    pub fn run_with_output(self) -> Result<Output> {
        if cfg!(windows) {
            run_with_output(self.windows, self.work_dir)
        } else {
            run_with_output(self.unix, self.work_dir)
        }
    }
}

pub fn run(cmdline: &str, dir: &str) -> Result<()> {
    do_run(cmdline, dir, |c| {
        c.stdout(Stdio::inherit());
    })
    .map(|_| ())
}

pub fn run_with_output(cmdline: &str, dir: &str) -> Result<Output> {
    do_run(cmdline, dir, |_| {})
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
        run(&format!("rustup run {} -- cargo fmt -- --check", TOOLCHAIN), ".")?;
    } else {
        run(&format!("rustup run {} -- cargo fmt", TOOLCHAIN), ".")?;
    }
    Ok(())
}

pub fn install_rustfmt() -> Result<()> {
    run(&format!("rustup install {}", TOOLCHAIN), ".")?;
    run(&format!("rustup component add rustfmt --toolchain {}", TOOLCHAIN), ".")
}

pub fn install_format_hook() -> Result<()> {
    let result_path = Path::new(if cfg!(windows) {
        "./.git/hooks/pre-commit.exe"
    } else {
        "./.git/hooks/pre-commit"
    });
    if !result_path.exists() {
        run("cargo build --package ra_tools --bin pre-commit", ".")?;
        if cfg!(windows) {
            fs::copy("./target/debug/pre-commit.exe", result_path)?;
        } else {
            fs::copy("./target/debug/pre-commit", result_path)?;
        }
    } else {
        Err(IoError::new(ErrorKind::AlreadyExists, "Git hook already created"))?;
    }
    Ok(())
}

pub fn run_clippy() -> Result<()> {
    match Command::new("rustup")
        .args(&["run", TOOLCHAIN, "--", "cargo", "clippy", "--version"])
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
    {
        Ok(status) if status.success() => (),
        _ => install_clippy()?,
    };

    let allowed_lints = [
        "clippy::collapsible_if",
        "clippy::map_clone", // FIXME: remove when Iterator::copied stabilizes (1.36.0)
        "clippy::needless_pass_by_value",
        "clippy::nonminimal_bool",
        "clippy::redundant_pattern_matching",
    ];
    run(
        &format!(
            "rustup run {} -- cargo clippy --all-features --all-targets -- -A {}",
            TOOLCHAIN,
            allowed_lints.join(" -A ")
        ),
        ".",
    )?;
    Ok(())
}

pub fn install_clippy() -> Result<()> {
    run(&format!("rustup install {}", TOOLCHAIN), ".")?;
    run(&format!("rustup component add clippy --toolchain {}", TOOLCHAIN), ".")
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

    run("rustup run nightly -- cargo fuzz run parser", "./crates/ra_syntax")
}

pub fn gen_tests(mode: Mode) -> Result<()> {
    let tests = tests_from_dir(&project_root().join(Path::new(GRAMMAR_DIR)))?;
    fn install_tests(tests: &HashMap<String, Test>, into: &str, mode: Mode) -> Result<()> {
        let tests_dir = project_root().join(into);
        if !tests_dir.is_dir() {
            fs::create_dir_all(&tests_dir)?;
        }
        // ok is never actually read, but it needs to be specified to create a Test in existing_tests
        let existing = existing_tests(&tests_dir, true)?;
        for t in existing.keys().filter(|&t| !tests.contains_key(t)) {
            panic!("Test is deleted: {}", t);
        }

        let mut new_idx = existing.len() + 1;
        for (name, test) in tests {
            let path = match existing.get(name) {
                Some((path, _test)) => path.clone(),
                None => {
                    let file_name = format!("{:04}_{}.rs", new_idx, name);
                    new_idx += 1;
                    tests_dir.join(file_name)
                }
            };
            teraron::update(&path, &test.text, mode)?;
        }
        Ok(())
    }
    install_tests(&tests.ok, OK_INLINE_TESTS_DIR, mode)?;
    install_tests(&tests.err, ERR_INLINE_TESTS_DIR, mode)
}

fn do_run<F>(cmdline: &str, dir: &str, mut f: F) -> Result<Output>
where
    F: FnMut(&mut Command),
{
    eprintln!("\nwill run: {}", cmdline);
    let proj_dir = project_root().join(dir);
    let mut args = cmdline.split_whitespace();
    let exec = args.next().unwrap();
    let mut cmd = Command::new(exec);
    f(cmd.args(args).current_dir(proj_dir).stderr(Stdio::inherit()));
    let output = cmd.output()?;
    if !output.status.success() {
        Err(format!("`{}` exited with {}", cmdline, output.status))?;
    }
    Ok(output)
}

#[derive(Default, Debug)]
struct Tests {
    pub ok: HashMap<String, Test>,
    pub err: HashMap<String, Test>,
}

fn tests_from_dir(dir: &Path) -> Result<Tests> {
    let mut res = Tests::default();
    for entry in ::walkdir::WalkDir::new(dir) {
        let entry = entry.unwrap();
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().unwrap_or_default() != "rs" {
            continue;
        }
        process_file(&mut res, entry.path())?;
    }
    let grammar_rs = dir.parent().unwrap().join("grammar.rs");
    process_file(&mut res, &grammar_rs)?;
    return Ok(res);
    fn process_file(res: &mut Tests, path: &Path) -> Result<()> {
        let text = fs::read_to_string(path)?;

        for (_, test) in collect_tests(&text) {
            if test.ok {
                if let Some(old_test) = res.ok.insert(test.name.clone(), test) {
                    Err(format!("Duplicate test: {}", old_test.name))?
                }
            } else {
                if let Some(old_test) = res.err.insert(test.name.clone(), test) {
                    Err(format!("Duplicate test: {}", old_test.name))?
                }
            }
        }
        Ok(())
    }
}

fn existing_tests(dir: &Path, ok: bool) -> Result<HashMap<String, (PathBuf, Test)>> {
    let mut res = HashMap::new();
    for file in fs::read_dir(dir)? {
        let file = file?;
        let path = file.path();
        if path.extension().unwrap_or_default() != "rs" {
            continue;
        }
        let name = {
            let file_name = path.file_name().unwrap().to_str().unwrap();
            file_name[5..file_name.len() - 3].to_string()
        };
        let text = fs::read_to_string(&path)?;
        let test = Test { name: name.clone(), text, ok };
        if let Some(old) = res.insert(name, (path, test)) {
            println!("Duplicate test: {:?}", old);
        }
    }
    Ok(res)
}
