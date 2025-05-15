use itertools::Itertools;
use rustc_lexer::{TokenKind, tokenize};
use shell_escape::escape;
use std::ffi::{OsStr, OsString};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::{fs, io};
use walkdir::WalkDir;

pub enum Error {
    CommandFailed(String, String),
    Io(io::Error),
    RustfmtNotInstalled,
    WalkDir(walkdir::Error),
    IntellijSetupActive,
    Parse(PathBuf, usize, String),
    CheckFailed,
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<walkdir::Error> for Error {
    fn from(error: walkdir::Error) -> Self {
        Self::WalkDir(error)
    }
}

impl Error {
    fn display(&self) {
        match self {
            Self::CheckFailed => {
                eprintln!("Formatting check failed!\nRun `cargo dev fmt` to update.");
            },
            Self::CommandFailed(command, stderr) => {
                eprintln!("error: command `{command}` failed!\nstderr: {stderr}");
            },
            Self::Io(err) => {
                eprintln!("error: {err}");
            },
            Self::RustfmtNotInstalled => {
                eprintln!("error: rustfmt nightly is not installed.");
            },
            Self::WalkDir(err) => {
                eprintln!("error: {err}");
            },
            Self::IntellijSetupActive => {
                eprintln!(
                    "error: a local rustc repo is enabled as path dependency via `cargo dev setup intellij`.\n\
                    Not formatting because that would format the local repo as well!\n\
                    Please revert the changes to `Cargo.toml`s with `cargo dev remove intellij`."
                );
            },
            Self::Parse(path, line, msg) => {
                eprintln!("error parsing `{}:{line}`: {msg}", path.display());
            },
        }
    }
}

struct FmtContext {
    check: bool,
    verbose: bool,
    rustfmt_path: String,
}

struct ClippyConf<'a> {
    name: &'a str,
    attrs: &'a str,
    lints: Vec<&'a str>,
    field: &'a str,
}

fn offset_to_line(text: &str, offset: usize) -> usize {
    match text.split('\n').try_fold((1usize, 0usize), |(line, pos), s| {
        let pos = pos + s.len() + 1;
        if pos > offset {
            ControlFlow::Break(line)
        } else {
            ControlFlow::Continue((line + 1, pos))
        }
    }) {
        ControlFlow::Break(x) | ControlFlow::Continue((x, _)) => x,
    }
}

/// Formats the configuration list in `clippy_config/src/conf.rs`
#[expect(clippy::too_many_lines)]
fn fmt_conf(check: bool) -> Result<(), Error> {
    #[derive(Clone, Copy)]
    enum State {
        Start,
        Docs,
        Pound,
        OpenBracket,
        Attr(u32),
        Lints,
        EndLints,
        Field,
    }

    let path = "clippy_config/src/conf.rs";
    let text = fs::read_to_string(path)?;

    let (pre, conf) = text
        .split_once("define_Conf! {\n")
        .expect("can't find config definition");
    let (conf, post) = conf.split_once("\n}\n").expect("can't find config definition");
    let conf_offset = pre.len() + 15;

    let mut pos = 0u32;
    let mut attrs_start = 0;
    let mut attrs_end = 0;
    let mut field_start = 0;
    let mut lints = Vec::new();
    let mut name = "";
    let mut fields = Vec::new();
    let mut state = State::Start;

    for (i, t) in tokenize(conf)
        .map(|x| {
            let start = pos;
            pos += x.len;
            (start as usize, x)
        })
        .filter(|(_, t)| !matches!(t.kind, TokenKind::Whitespace))
    {
        match (state, t.kind) {
            (State::Start, TokenKind::LineComment { doc_style: Some(_) }) => {
                attrs_start = i;
                attrs_end = i + t.len as usize;
                state = State::Docs;
            },
            (State::Start, TokenKind::Pound) => {
                attrs_start = i;
                attrs_end = i;
                state = State::Pound;
            },
            (State::Docs, TokenKind::LineComment { doc_style: Some(_) }) => attrs_end = i + t.len as usize,
            (State::Docs, TokenKind::Pound) => state = State::Pound,
            (State::Pound, TokenKind::OpenBracket) => state = State::OpenBracket,
            (State::OpenBracket, TokenKind::Ident) => {
                state = if conf[i..i + t.len as usize] == *"lints" {
                    State::Lints
                } else {
                    State::Attr(0)
                };
            },
            (State::Attr(0), TokenKind::CloseBracket) => {
                attrs_end = i + 1;
                state = State::Docs;
            },
            (State::Attr(x), TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::OpenBrace) => {
                state = State::Attr(x + 1);
            },
            (State::Attr(x), TokenKind::CloseParen | TokenKind::CloseBracket | TokenKind::CloseBrace) => {
                state = State::Attr(x - 1);
            },
            (State::Lints, TokenKind::Ident) => lints.push(&conf[i..i + t.len as usize]),
            (State::Lints, TokenKind::CloseBracket) => state = State::EndLints,
            (State::EndLints | State::Docs, TokenKind::Ident) => {
                field_start = i;
                name = &conf[i..i + t.len as usize];
                state = State::Field;
            },
            (State::Field, TokenKind::LineComment { doc_style: Some(_) }) => {
                #[expect(clippy::drain_collect)]
                fields.push(ClippyConf {
                    name,
                    attrs: &conf[attrs_start..attrs_end],
                    lints: lints.drain(..).collect(),
                    field: conf[field_start..i].trim_end(),
                });
                attrs_start = i;
                attrs_end = i + t.len as usize;
                state = State::Docs;
            },
            (State::Field, TokenKind::Pound) => {
                #[expect(clippy::drain_collect)]
                fields.push(ClippyConf {
                    name,
                    attrs: &conf[attrs_start..attrs_end],
                    lints: lints.drain(..).collect(),
                    field: conf[field_start..i].trim_end(),
                });
                attrs_start = i;
                attrs_end = i;
                state = State::Pound;
            },
            (State::Field | State::Attr(_), _)
            | (State::Lints, TokenKind::Comma | TokenKind::OpenParen | TokenKind::CloseParen) => {},
            _ => {
                return Err(Error::Parse(
                    PathBuf::from(path),
                    offset_to_line(&text, conf_offset + i),
                    format!("unexpected token `{}`", &conf[i..i + t.len as usize]),
                ));
            },
        }
    }

    if !matches!(state, State::Field) {
        return Err(Error::Parse(
            PathBuf::from(path),
            offset_to_line(&text, conf_offset + conf.len()),
            "incomplete field".into(),
        ));
    }
    fields.push(ClippyConf {
        name,
        attrs: &conf[attrs_start..attrs_end],
        lints,
        field: conf[field_start..].trim_end(),
    });

    for field in &mut fields {
        field.lints.sort_unstable();
    }
    fields.sort_by_key(|x| x.name);

    let new_text = format!(
        "{pre}define_Conf! {{\n{}}}\n{post}",
        fields.iter().format_with("", |field, f| {
            if field.lints.is_empty() {
                f(&format_args!("    {}\n    {}\n", field.attrs, field.field))
            } else if field.lints.iter().map(|x| x.len() + 2).sum::<usize>() < 120 - 14 {
                f(&format_args!(
                    "    {}\n    #[lints({})]\n    {}\n",
                    field.attrs,
                    field.lints.iter().join(", "),
                    field.field,
                ))
            } else {
                f(&format_args!(
                    "    {}\n    #[lints({}\n    )]\n    {}\n",
                    field.attrs,
                    field
                        .lints
                        .iter()
                        .format_with("", |x, f| f(&format_args!("\n        {x},"))),
                    field.field,
                ))
            }
        })
    );

    if text != new_text {
        if check {
            return Err(Error::CheckFailed);
        }
        fs::write(path, new_text.as_bytes())?;
    }
    Ok(())
}

fn run_rustfmt(context: &FmtContext) -> Result<(), Error> {
    // if we added a local rustc repo as path dependency to clippy for rust analyzer, we do NOT want to
    // format because rustfmt would also format the entire rustc repo as it is a local
    // dependency
    if fs::read_to_string("Cargo.toml")
        .expect("Failed to read clippy Cargo.toml")
        .contains("[target.'cfg(NOT_A_PLATFORM)'.dependencies]")
    {
        return Err(Error::IntellijSetupActive);
    }

    check_for_rustfmt(context)?;

    cargo_fmt(context, ".".as_ref())?;
    cargo_fmt(context, "clippy_dev".as_ref())?;
    cargo_fmt(context, "rustc_tools_util".as_ref())?;
    cargo_fmt(context, "lintcheck".as_ref())?;

    let chunks = WalkDir::new("tests")
        .into_iter()
        .filter_map(|entry| {
            let entry = entry.expect("failed to find tests");
            let path = entry.path();
            if path.extension() != Some("rs".as_ref())
                || path
                    .components()
                    .nth_back(1)
                    .is_some_and(|c| c.as_os_str() == "syntax-error-recovery")
                || entry.file_name() == "ice-3891.rs"
            {
                None
            } else {
                Some(entry.into_path().into_os_string())
            }
        })
        .chunks(250);

    for chunk in &chunks {
        rustfmt(context, chunk)?;
    }
    Ok(())
}

// the "main" function of cargo dev fmt
pub fn run(check: bool, verbose: bool) {
    let output = Command::new("rustup")
        .args(["which", "rustfmt"])
        .stderr(Stdio::inherit())
        .output()
        .expect("error running `rustup which rustfmt`");
    if !output.status.success() {
        eprintln!("`rustup which rustfmt` did not execute successfully");
        process::exit(1);
    }
    let mut rustfmt_path = String::from_utf8(output.stdout).expect("invalid rustfmt path");
    rustfmt_path.truncate(rustfmt_path.trim_end().len());

    let context = FmtContext {
        check,
        verbose,
        rustfmt_path,
    };
    if let Err(e) = run_rustfmt(&context).and_then(|()| fmt_conf(check)) {
        e.display();
        process::exit(1);
    }
}

fn format_command(program: impl AsRef<OsStr>, dir: impl AsRef<Path>, args: &[impl AsRef<OsStr>]) -> String {
    let arg_display: Vec<_> = args.iter().map(|a| escape(a.as_ref().to_string_lossy())).collect();

    format!(
        "cd {} && {} {}",
        escape(dir.as_ref().to_string_lossy()),
        escape(program.as_ref().to_string_lossy()),
        arg_display.join(" ")
    )
}

fn exec_fmt_command(
    context: &FmtContext,
    program: impl AsRef<OsStr>,
    dir: impl AsRef<Path>,
    args: &[impl AsRef<OsStr>],
) -> Result<(), Error> {
    if context.verbose {
        println!("{}", format_command(&program, &dir, args));
    }

    let output = Command::new(&program)
        .env("RUSTFMT", &context.rustfmt_path)
        .current_dir(&dir)
        .args(args.iter())
        .output()
        .unwrap();
    let success = output.status.success();

    match (context.check, success) {
        (_, true) => Ok(()),
        (true, false) => Err(Error::CheckFailed),
        (false, false) => {
            let stderr = std::str::from_utf8(&output.stderr).unwrap_or("");
            Err(Error::CommandFailed(
                format_command(&program, &dir, args),
                String::from(stderr),
            ))
        },
    }
}

fn cargo_fmt(context: &FmtContext, path: &Path) -> Result<(), Error> {
    let mut args = vec!["fmt", "--all"];
    if context.check {
        args.push("--check");
    }
    exec_fmt_command(context, "cargo", path, &args)
}

fn check_for_rustfmt(context: &FmtContext) -> Result<(), Error> {
    let program = "rustfmt";
    let dir = std::env::current_dir()?;
    let args = &["--version"];

    if context.verbose {
        println!("{}", format_command(program, &dir, args));
    }

    let output = Command::new(program).current_dir(&dir).args(args.iter()).output()?;

    if output.status.success() {
        Ok(())
    } else if std::str::from_utf8(&output.stderr)
        .unwrap_or("")
        .starts_with("error: 'rustfmt' is not installed")
    {
        Err(Error::RustfmtNotInstalled)
    } else {
        Err(Error::CommandFailed(
            format_command(program, &dir, args),
            std::str::from_utf8(&output.stderr).unwrap_or("").to_string(),
        ))
    }
}

fn rustfmt(context: &FmtContext, paths: impl Iterator<Item = OsString>) -> Result<(), Error> {
    let mut args = Vec::new();
    if context.check {
        args.push(OsString::from("--check"));
    }
    args.extend(paths);
    exec_fmt_command(context, &context.rustfmt_path, std::env::current_dir()?, &args)
}
