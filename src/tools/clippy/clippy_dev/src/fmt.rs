use crate::utils::{
    ClippyInfo, ErrAction, FileUpdater, UpdateMode, UpdateStatus, panic_action, run_with_args_split, run_with_output,
};
use itertools::Itertools;
use rustc_lexer::{TokenKind, tokenize};
use std::fmt::Write;
use std::fs;
use std::io::{self, Read};
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::{self, Command, Stdio};
use walkdir::WalkDir;

pub enum Error {
    Io(io::Error),
    Parse(PathBuf, usize, String),
    CheckFailed,
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl Error {
    fn display(&self) {
        match self {
            Self::CheckFailed => {
                eprintln!("Formatting check failed!\nRun `cargo dev fmt` to update.");
            },
            Self::Io(err) => {
                eprintln!("error: {err}");
            },
            Self::Parse(path, line, msg) => {
                eprintln!("error parsing `{}:{line}`: {msg}", path.display());
            },
        }
    }
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

/// Format the symbols list
fn fmt_syms(update_mode: UpdateMode) {
    FileUpdater::default().update_file_checked(
        "cargo dev fmt",
        update_mode,
        "clippy_utils/src/sym.rs",
        &mut |_, text: &str, new_text: &mut String| {
            let (pre, conf) = text.split_once("generate! {\n").expect("can't find generate! call");
            let (conf, post) = conf.split_once("\n}\n").expect("can't find end of generate! call");
            let mut lines = conf
                .lines()
                .map(|line| {
                    let line = line.trim();
                    line.strip_suffix(',').unwrap_or(line).trim_end()
                })
                .collect::<Vec<_>>();
            lines.sort_unstable();
            write!(
                new_text,
                "{pre}generate! {{\n    {},\n}}\n{post}",
                lines.join(",\n    "),
            )
            .unwrap();
            if text == new_text {
                UpdateStatus::Unchanged
            } else {
                UpdateStatus::Changed
            }
        },
    );
}

fn run_rustfmt(clippy: &ClippyInfo, update_mode: UpdateMode) {
    let mut rustfmt_path = String::from_utf8(run_with_output(
        "rustup which rustfmt",
        Command::new("rustup").args(["which", "rustfmt"]),
    ))
    .expect("invalid rustfmt path");
    rustfmt_path.truncate(rustfmt_path.trim_end().len());

    let mut cargo_path = String::from_utf8(run_with_output(
        "rustup which cargo",
        Command::new("rustup").args(["which", "cargo"]),
    ))
    .expect("invalid cargo path");
    cargo_path.truncate(cargo_path.trim_end().len());

    // Start all format jobs first before waiting on the results.
    let mut children = Vec::with_capacity(16);
    for &path in &[
        ".",
        "clippy_config",
        "clippy_dev",
        "clippy_lints",
        "clippy_lints_internal",
        "clippy_utils",
        "rustc_tools_util",
        "lintcheck",
    ] {
        let mut cmd = Command::new(&cargo_path);
        cmd.current_dir(clippy.path.join(path))
            .args(["fmt"])
            .env("RUSTFMT", &rustfmt_path)
            .stdout(Stdio::null())
            .stdin(Stdio::null())
            .stderr(Stdio::piped());
        if update_mode.is_check() {
            cmd.arg("--check");
        }
        match cmd.spawn() {
            Ok(x) => children.push(("cargo fmt", x)),
            Err(ref e) => panic_action(&e, ErrAction::Run, "cargo fmt".as_ref()),
        }
    }

    run_with_args_split(
        || {
            let mut cmd = Command::new(&rustfmt_path);
            if update_mode.is_check() {
                cmd.arg("--check");
            }
            cmd.stdout(Stdio::null())
                .stdin(Stdio::null())
                .stderr(Stdio::piped())
                .args(["--config", "show_parse_errors=false"]);
            cmd
        },
        |cmd| match cmd.spawn() {
            Ok(x) => children.push(("rustfmt", x)),
            Err(ref e) => panic_action(&e, ErrAction::Run, "rustfmt".as_ref()),
        },
        WalkDir::new("tests")
            .into_iter()
            .filter_entry(|p| p.path().file_name().is_none_or(|x| x != "skip_rustfmt"))
            .filter_map(|e| {
                let e = e.expect("error reading `tests`");
                e.path()
                    .as_os_str()
                    .as_encoded_bytes()
                    .ends_with(b".rs")
                    .then(|| e.into_path().into_os_string())
            }),
    );

    for (name, child) in &mut children {
        match child.wait() {
            Ok(status) => match (update_mode, status.exit_ok()) {
                (UpdateMode::Check | UpdateMode::Change, Ok(())) => {},
                (UpdateMode::Check, Err(_)) => {
                    let mut s = String::new();
                    if let Some(mut stderr) = child.stderr.take()
                        && stderr.read_to_string(&mut s).is_ok()
                    {
                        eprintln!("{s}");
                    }
                    eprintln!("Formatting check failed!\nRun `cargo dev fmt` to update.");
                    process::exit(1);
                },
                (UpdateMode::Change, Err(e)) => {
                    let mut s = String::new();
                    if let Some(mut stderr) = child.stderr.take()
                        && stderr.read_to_string(&mut s).is_ok()
                    {
                        eprintln!("{s}");
                    }
                    panic_action(&e, ErrAction::Run, name.as_ref());
                },
            },
            Err(ref e) => panic_action(e, ErrAction::Run, name.as_ref()),
        }
    }
}

// the "main" function of cargo dev fmt
pub fn run(clippy: &ClippyInfo, update_mode: UpdateMode) {
    if clippy.has_intellij_hook {
        eprintln!(
            "error: a local rustc repo is enabled as path dependency via `cargo dev setup intellij`.\n\
            Not formatting because that would format the local repo as well!\n\
            Please revert the changes to `Cargo.toml`s with `cargo dev remove intellij`."
        );
        return;
    }
    run_rustfmt(clippy, update_mode);
    fmt_syms(update_mode);
    if let Err(e) = fmt_conf(update_mode.is_check()) {
        e.display();
        process::exit(1);
    }
}
